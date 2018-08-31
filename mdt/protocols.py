import collections
import glob
import numbers
import os
from warnings import warn

import numpy as np
import copy

from mdt.lib.exceptions import ProtocolIOError

__author__ = 'Robbert Harms'
__date__ = "2014-02-06"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Protocol(collections.Mapping):

    def __init__(self, columns=None):
        """Create a new protocol. Optionally initializes the protocol with the given set of columns.

        Please note that we use SI units throughout MDT. Take care when loading the data that you load it in SI units.

        For example:

        * G (gradient amplitude) in T/m (Tesla per meter)
        * Delta (time interval) in seconds
        * delta (gradient duration) in seconds

        Args:
            columns (dict): The initial list of columns used by this protocol, the keys should be the name of the
                parameter (the same as those used in the model functions).
                The values should be numpy arrays of equal length.
        """
        super().__init__()
        self._gamma_h = 267.5987E6 # radians s^-1 T^-1 (s = seconds, T = Tesla)
        self._unweighted_threshold = 25e6 # s/m^2
        self._columns = {}
        self._preferred_column_order = ('gx', 'gy', 'gz', 'G', 'Delta', 'delta', 'TE', 'T1', 'b', 'q', 'maxG')
        self._virtual_columns = [VirtualColumnB(),
                                 SimpleVirtualColumn('Delta', lambda protocol: get_sequence_timings(protocol)['Delta']),
                                 SimpleVirtualColumn('delta', lambda protocol: get_sequence_timings(protocol)['delta']),
                                 SimpleVirtualColumn('G', lambda protocol: get_sequence_timings(protocol)['G'])]

        if columns:
            if 'g' in columns:
                columns.update({'gx': columns['g'][:, 0], 'gy': columns['g'][:, 1], 'gz': columns['g'][:, 2]})
                del columns['g']

            self._columns = columns

            for k, v in columns.items():
                s = v.shape

                if len(s) > 2 or (len(s) == 2 and s[1] > 1):
                    raise ValueError("All columns should be of width one.")

                if len(s) ==2 and s[1] == 1:
                    self._columns[k] = np.squeeze(v)

    @property
    def gamma_h(self):
        """Get the used gamma of the ``H`` atom used by this protocol.

        Returns:
            float: The used gamma of the ``H`` atom used by this protocol.
        """
        return self._gamma_h

    def with_updates(self, additional_columns):
        """Creates a copy of this protocol with the given columns added.

        Args:
            additional_columns (dict): the additional columns to add

        Returns:
            Protocol: the new updated protocol
        """
        protocol = self
        for key, value in additional_columns.items():
            protocol = protocol.with_update(key, value)
        return protocol

    def with_update(self, name, data):
        """Create a copy of the protocol with the given column updated to a new value.

        Synonymous to :meth:`with_new_column`.

        Args:
            name (str): The name of the column to add
            data (ndarray or float): The value or vector to add to this protocol.

        Returns:
            Protocol: the updated protocol
        """
        return self.with_new_column(name, data)

    def with_new_column(self, name, data):
        """Create a copy of this protocol with the given column updated/added.

        Args:
            name (str): The name of the column to add
            data (ndarray): The vector to add to this protocol.

        Returns:
            Protocol: the new protocol with the updated columns
        """
        columns = copy.copy(self._columns)

        if isinstance(data, str):
            data = float(data)

        if isinstance(data, numbers.Number) or not data.shape:
            if self._columns:
                data = np.ones((self.length,)) * data
            else:
                data = np.ones((1,)) * data

        s = data.shape
        if self.length and s[0] > self.length:
            warn("The column '{}' has to many elements ({}), we will only use the first {}.".format(
                name, s[0], self.length))
            columns.update({name: data[:self.length]})
        elif self.length and s[0] < self.length:
            raise ValueError("Incorrect column length given for '{}', expected {} and got {}.".format(
                name, self.length, s[0]))
        else:
            if name == 'g' and len(data.shape) > 1 and data.shape[1] == 3:
                columns.update({'gx': data[:, 0], 'gy': data[:, 1], 'gz': data[:, 2]})
            else:
                columns.update({name: data})
        return Protocol(columns)

    def with_added_column_from_file(self, name, file_name, multiplication_factor=1):
        """Create a copy of this protocol with the given column (loaded from a file) added to this protocol.

        The given file can either contain a single value or one value per protocol line.

        Args:
            name (str): The name of the column to add.
            file_name (str): The file to get the column from.
            multiplication_factor (double): we might need to scale the data by a constant. For example,
                if the data in the file is in ms we might need to scale it to seconds by multiplying with 1e-3
        Returns:
            self: for chaining
        """
        columns = copy.copy(self._columns)
        if name == 'g':
            columns.update(get_g_columns(file_name))
            for column_name in ('gx', 'gy', 'gz'):
                columns[column_name] *= multiplication_factor
            return Protocol(columns)
        else:
            data = np.genfromtxt(file_name)
            data *= multiplication_factor
            return self.with_new_column(name, data)

    def with_column_removed(self, column_name):
        """Create a copy of this protocol with the given column removed.

        Args:
            column_name (str): The name of the column to remove

        Returns:
            Protocol: the new updated protocol
        """
        columns = copy.copy(self._columns)

        if column_name == 'g':
            del columns['gx']
            del columns['gy']
            del columns['gz']
        else:
            if column_name in self._columns:
                del columns[column_name]

        return Protocol(columns)

    def with_columns_removed(self, column_names):
        """Create a copy of this protocol with the given columns removed.

        Args:
            column_names (list of str): The name of the columns to remove

        Returns:
            Protocol: the new updated protocol
        """
        columns = copy.copy(self._columns)

        for column_name in column_names:
            if column_name == 'g':
                del columns['gx']
                del columns['gy']
                del columns['gz']
            else:
                if column_name in self._columns:
                    del columns[column_name]

        return Protocol(columns)

    def with_rows_removed(self, rows):
        """Create a copy of the protocol with a list of rows removed from all the columns.

        Please note that the protocol is 0 indexed.

        Args:
            rows (list of int): List with indices of the rows to remove
        """
        columns = copy.copy(self._columns)
        for key, column in columns.items():
            columns[key] = np.delete(column, rows)
        return Protocol(columns)

    def append_protocol(self, protocol):
        """Append another protocol to this protocol and return the result as a new protocol.

        This will add the columns of the other protocol to the columns of (a copy of) this protocol
        This supposes that both protocols have the same columns.
        """
        columns = copy.copy(self._columns)
        if type(protocol) is type(self):
            for key in self._columns:
                columns[key] = np.append(self[key], protocol[key], 0)
        return Protocol(columns)

    def get_columns(self, column_names):
        """Get a matrix containing the requested column names in the order given.

        Returns:
            ndarrray: A 2d matrix with the column requested concatenated.
        """
        if not column_names:
            return None
        return np.concatenate([self[i] for i in column_names], axis=1)

    def get_column(self, column_name):
        """Get the column associated by the given column name.

        Args:
            column_name (str): The name of the column we want to return.

        Returns:
            ndarray: The column we would like to return. This is returned as a 2d matrix with shape (n, 1).

        Raises:
            KeyError: If the column could not be found.
        """
        try:
            return self._get_real_column(column_name)
        except KeyError:
            try:
                return self._get_estimated_column(column_name)
            except KeyError:
                raise KeyError('The given column name "{}" could not be found in this protocol.'.format(column_name))

    @property
    def length(self):
        """Get the length of this protocol.

        Returns:
            int: The length of the protocol.
        """
        if self._columns:
            return self._columns[list(self._columns.keys())[0]].shape[0]
        return 0

    @property
    def number_of_columns(self):
        """Get the number of columns in this protocol.

        This only counts the real columns, not the estimated ones.

        Returns:
            int: The number columns in this protocol.
        """
        return len(self._columns)

    @property
    def column_names(self):
        """Get the names of the columns.

        This only lists the real columns, not the estimated ones.

        Returns:
            list of str: The names of the columns.
        """
        return self._column_names_in_preferred_order(self._columns.keys())

    @property
    def estimated_column_names(self):
        """Get the names of the virtual columns.

        This will only return the names of the virtual columns for which no real column exists.
        """
        return self._column_names_in_preferred_order([e.name for e in self._virtual_columns if
                                                      e.name not in self.column_names])

    def get_nmr_shells(self):
        """Get the number of unique shells in this protocol.

        This is measured by counting the number of unique weighted bvals in this protocol.

        Returns:
            int: The number of unique weighted b-values in this protocol

        Raises:
            KeyError: This function may throw a key error if the 'b' column in the protocol could not be loaded.
        """
        return len(self.get_b_values_shells())

    def get_b_values_shells(self, width=0.1e9):
        """Get the b-values of the unique shells in this protocol.

        Args:
            width (float): assume a certain bandwidth of b-values around each shell. This will group b-values together
                if they are not more than

        Returns:
            list: per b-value the information about that shell as a dictionary. Each of these dicts contains the
                ``b_value`` and the ``nmr_volumes`` keys.

        Raises:
            KeyError: This function may throw a key error if the 'b' column in the protocol could not be loaded.
        """
        def cluster_b_values(b_values):
            clustered_shells = []
            new_shell = []
            for ind in range(len(b_values) - 1):
                new_shell.append(b_values[ind])
                if b_values[ind + 1] - b_values[ind] > width:
                    clustered_shells.append(new_shell)
                    new_shell = []

            if new_shell:
                clustered_shells.append(new_shell)
            return clustered_shells

        clusters = cluster_b_values(np.sort(self.get_column('b')[self.get_weighted_indices()]))

        cluster_info = []
        for cluster in clusters:
            cluster_info.append({'b_value': np.mean(cluster), 'nmr_volumes': len(cluster)})

        return cluster_info

    def has_column(self, column_name):
        """Check if this protocol has a column with the given name.

        This will also return true if the column can be estimated from the other columns. See is_column_real()
        to get information for columns that are really known.

        Returns:
            boolean: true if there is a column with the given name, false otherwise.
        """
        try:
            return self.get_column(column_name) is not None
        except KeyError:
            return False

    def is_column_real(self, column_name):
        """Check if this protocol has real column information for the column with the given name.

        For example, the other function has_column('G') will normally return true since 'G' can be estimated from 'b'.
        This function will return false if the column needs to be estimated and will return true if real data
        is available for the columnn.

        Returns:
            boolean: true if there is really a column with the given name, false otherwise.
        """
        return column_name in self._columns

    def get_unweighted_indices(self, unweighted_threshold=None):
        """Get the indices to the unweighted volumes.

        If the column 'b' could not be found, assume that all measurements are unweighted.

        Args:
            unweighted_threshold (float): the threshold under which we call it unweighted.

        Returns:
            list of int: A list of indices to the unweighted volumes.
        """
        unweighted_threshold = unweighted_threshold or self._unweighted_threshold

        try:
            b = self.get_column('b')
            g = self.get_column('g')
            return np.where(np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2) * b[:, 0] < unweighted_threshold)[0]
        except KeyError:
            return range(self.length)

    def get_weighted_indices(self, unweighted_threshold=None):
        """Get the indices to the weighted volumes.

        Args:
            unweighted_threshold (float): the threshold under which we call it unweighted.

        Returns:
            list of int: A list of indices to the weighted volumes.
        """
        return sorted(set(range(self.get_column('b').shape[0])) -
                      set(self.get_unweighted_indices(unweighted_threshold=unweighted_threshold)))

    def get_indices_bval_in_range(self, start=0, end=1.0e9):
        """Get the indices of the b-values in the range [start, end].

        This can be used to get the indices of gradients whose b-value is in the range suitable for
        a specific analysis.

        Note that we use SI units and you need to specify the values in units of s/m^2 and not in s/mm^2.

        Also note that specifying 0 as start of the range does not automatically mean that the unweighted volumes are
        returned. It can happen that the b-value of the unweighted volumes is higher then 0 even if the the gradient
        ``g`` is ``[0 0 0]``. This function does not make any assumptions about that and just returns indices in the
        given range.

        If you want to include the unweighted volumes, make a call to :meth:`get_unweighted_indices` yourself.

        Args:
            start (float): b-value of the start of the range (inclusive) we want to get the indices of the volumes from.
                Should be positive. We subtract epsilon for float comparison
            end (float): b-value of the end of the range (inclusive) we want to get the indices of the volumes from.
                Should be positive. We add epsilon for float comparison
            epsilon (float): the epsilon we use in the range.

        Returns:
            :class:`list`: a list of indices of all volumes whose b-value is in the given range.
                If you want to include the unweighted volumes, make a call to get_unweighted_indices() yourself.
        """
        b_values = self.get_column('b')
        return np.where((start <= b_values) * (b_values <= end))[0]

    def get_all_columns(self):
        """Get all real (known) columns as a big array.

        Returns:
            ndarray: All the real columns of this protocol.
        """
        return self.get_columns(self.column_names)

    def deepcopy(self):
        """Return a deep copy of this protocol.

        Returns:
            Protocol: A deep copy of this protocol.
        """
        return Protocol(columns=copy.deepcopy(self._columns))

    def get_new_protocol_with_indices(self, indices):
        """Create a new protocol object with all the columns but as rows only those of the given indices.

        Args:
            indices: the indices we want to use in the new protocol

        Returns:
            Protocol: a protocol with all the data of the given indices
        """
        return Protocol(columns={k: v[indices] for k, v in self._columns.items()})

    def _get_real_column(self, column_name):
        """Try to use a real column from this protocol.

        Returns:
            ndarray: A real column, that is, a column from which we have real data.

        Raises:
            KeyError: If the column name could not be found we raise a key error.
        """
        if column_name in self._columns:
            return np.reshape(self._columns[column_name], (-1, 1))

        if column_name == 'g':
            return self.get_columns(('gx', 'gy', 'gz'))

        raise KeyError('The given column could not be found.')

    def _get_estimated_column(self, column_name):
        """Try to use an estimated column from this protocol.

        This uses the list of virtual columns to try to estimate the requested column.

        Returns:
            ndarray: An estimated column, that is, a column we estimate from the other columns.

        Raises:
            KeyError: If the column name could not be estimated we raise a key error.
        """
        for virtual_column in self._virtual_columns:
            if virtual_column.name == column_name:
                return np.reshape(virtual_column.get_values(self), (-1, 1))

        raise KeyError('The given column name "{}" could not be found in this protocol.'.format(column_name))

    def _column_names_in_preferred_order(self, column_names):
        """Sort the given column names in the preferred order.

        Column names not in the list of preferred ordering are appended to the end of the list.

        Args:
            column_names (list): the list of column names

        Returns:
            list: the list of column names in the preferred order
        """
        columns_list = [n for n in column_names]
        final_list = []
        for column_name in self._preferred_column_order:
            if column_name in columns_list:
                columns_list.remove(column_name)
                final_list.append(column_name)

        final_list.extend(columns_list)
        return final_list

    def __len__(self):
        return self.length

    def __contains__(self, column):
        return self.has_column(column)

    def __getitem__(self, column):
        return self.get_column(column)

    def __iter__(self):
        for key in self._columns.keys():
            yield key

    def __str__(self):
        s = 'Column names: ' + ', '.join(self.column_names) + "\n"
        s += 'Data: ' + "\n"
        s += np.array_str(self.get_all_columns())
        return s


class VirtualColumn:

    def __init__(self, name):
        """The interface for generating virtual columns.

        Virtual columns are columns generated on the fly from the other parts of the protocol. They are
        generally only generated if the column it tries to generate is not in the protocol.

        In the Protocol they are used separately from the RealColumns. The VirtualColumns can always be added to
        the Protocol, but are only used when needed. The RealColumns can overrule VirtualColumns by their presence.

        Args:
            name (str): the name of the column this object generates.
        """
        self.name = name

    def get_values(self, parent_protocol):
        """Get the column given the information in the given protocol.

        Args:
            parent_protocol (Protocol): the protocol object to use as a basis for generating the column

        Returns:
            ndarray: the single column as a row vector or 2d matrix of shape nx1
        """


class SimpleVirtualColumn(VirtualColumn):

    def __init__(self, name, generate_function):
        """Create a simple virtual column that uses the given generate function to get the column.

        Args:
            name (str): the name of the column
            generate_function (python function): the function to generate the column
        """
        super().__init__(name)
        self._generate_function = generate_function

    def get_values(self, parent_protocol):
        return self._generate_function(parent_protocol)


class VirtualColumnB(VirtualColumn):

    def __init__(self):
        super().__init__('b')

    def get_values(self, parent_protocol):
        sequence_timings = get_sequence_timings(parent_protocol)
        return np.reshape(np.array(parent_protocol.gamma_h ** 2 *
                                   sequence_timings['G'] ** 2 *
                                   sequence_timings['delta'] ** 2 *
                                   (sequence_timings['Delta'] - (sequence_timings['delta'] / 3))), (-1, 1))


def get_sequence_timings(protocol):
    """Return G, Delta and delta, estimate them if necessary.

    If Delta and delta are available, they are used instead of estimated Delta and delta.

    Args:
        protocol (Protocol): the protocol for which we want to get the sequence timings.

    Returns:
        dict: the columns G, Delta and delta
    """
    def all_real(columns):
        return all(map(protocol.is_column_real, columns))

    if all_real(['G', 'delta', 'Delta']):
        return {name: protocol[name] for name in ['G', 'delta', 'Delta']}

    if all_real(['b', 'Delta', 'delta']):
        G = np.sqrt(protocol['b'] / (protocol.gamma_h ** 2 * protocol['delta'] ** 2 *
                                     (protocol['Delta'] - (protocol['delta'] / 3.0))))
        G[protocol.get_unweighted_indices()] = 0
        return {'G': G, 'Delta': protocol['Delta'], 'delta': protocol['delta']}

    if all_real(['b', 'Delta', 'G']):
        input_array = np.zeros((protocol.length, 4))
        input_array[:, 0] = -1 / 3.0
        input_array[:, 1] = np.squeeze(protocol['Delta'])
        input_array[:, 2] = 0
        input_array[:, 3] = np.squeeze(-protocol['b'] / (protocol.gamma_h ** 2 * protocol['G'] ** 2))

        b = protocol['b']
        delta = np.zeros((protocol.length, 1))

        for ind in range(protocol.length):
            if b[ind] == 0:
                delta[ind] = 0
            else:
                roots = np.roots(input_array[ind])
                delta[ind] = roots[0]

        return {'G': protocol['G'], 'Delta': protocol['Delta'], 'delta': delta}

    if all_real(['b', 'G', 'delta']):
        Delta = np.nan_to_num(np.array((protocol['b'] - protocol.gamma_h ** 2 * protocol['G'] ** 2
                                        * protocol['delta'] ** 3 / 3.0) /
                                       (protocol.gamma_h ** 2 * protocol['G'] ** 2 * protocol['delta'] ** 2)))
        return {'G': protocol['G'], 'delta': protocol['delta'], 'Delta': Delta}

    if not protocol.is_column_real('b'):
        raise KeyError('Can not estimate the sequence timings, column "b" is not provided.')

    if protocol.has_column('maxG'):
        maxG = protocol['maxG']
    else:
        maxG = np.reshape(np.ones((protocol.length,)) * 0.04, (-1, 1))

    bvals = protocol['b']
    shells = [shell['b_value'] for shell in protocol.get_b_values_shells()]
    if shells:
        bmax = max(shells)
    else:
        bmax = 1

    Deltas = (3 * bmax / (2 * protocol.gamma_h ** 2 * maxG ** 2)) ** (1 / 3.0)
    deltas = Deltas
    G = np.sqrt(bvals / bmax) * maxG

    return {'G': G, 'Delta': Deltas, 'delta': deltas}


def load_bvec_bval(bvec, bval, column_based='auto', bval_scale='auto'):
    """Load an protocol from a bvec and bval file.

    This supposes that the bvec (the vector file) has 3 rows (gx, gy, gz) and is space or tab seperated.
    The bval file (the b values) are one one single line with space or tab separated b values.

    Args:
        bvec (str): The filename of the bvec file
        bval (str): The filename of the bval file
        column_based (boolean): If true, this supposes that the bvec (the vector file) has 3 rows (gx, gy, gz)
            and is space or tab seperated and that the bval file (the b values) are one one single line with space or
            tab separated b values. If false, the vectors and b values are each one a different line.
            If 'auto' it is autodetected, this is the default.
        bval_scale (float): The amount by which we want to scale (multiply) the b-values. Typically bval files are
            in units of s/mm^2, while MDT uses s/m^2 in computations. To rescale, this function checks if the b-val is
            lower then 1e4 and if so multiplies it by 1e6.

    Returns:
        Protocol the loaded protocol.
    """
    bvec = get_g_columns(bvec, column_based=column_based)
    bval = np.expand_dims(np.genfromtxt(bval), axis=1)

    if bval_scale == 'auto' and bval[0, 0] < 1e4:
        bval *= 1e6
    else:
        bval *= bval_scale

    columns = {'b': bval}
    columns.update(bvec)

    if bvec['gx'].shape[0] != bval.shape[0]:
        raise ValueError('Columns not of same length.')

    return Protocol(columns=columns)


def get_g_columns(bvec_file, column_based='auto'):
    """Get the columns of a bvec file. Use auto transpose if needed.

    Args:
        bvec_file (str): The filename of the bvec file
        column_based (boolean): If true, this supposes that the bvec (the vector file) has 3 rows (gx, gy, gz)
            and is space or tab seperated
            If false, the vectors are each one a different line.
            If 'auto' it is autodetected, this is the default.

    Returns:
        dict: the loaded bvec matrix separated into 'gx', 'gy' and 'gz'
    """
    bvec = np.genfromtxt(bvec_file)

    if len(bvec.shape) < 2:
        raise ValueError('Bvec file does not have enough dimensions.')

    if column_based == 'auto':
        if bvec.shape[1] > bvec.shape[0]:
            bvec = bvec.transpose()
    elif column_based:
        bvec = bvec.transpose()

    return {'gx': np.reshape(bvec[:, 0], (-1, 1)),
            'gy': np.reshape(bvec[:, 1], (-1, 1)),
            'gz': np.reshape(bvec[:, 2], (-1, 1))}


def write_bvec_bval(protocol, bvec_fname, bval_fname, column_based=True, bval_scale=1):
    """Write the given protocol to bvec and bval files.

    This writes the bvector and bvalues to the given filenames.

    Args:
        protocol (Protocol): The protocol to write to bvec and bval files.
        bvec_fname (string): The bvector filename
        bval_fname (string): The bval filename
        column_based (boolean, optional, default true):
            If true, this supposes that the bvec (the vector file) will have 3 rows (gx, gy, gz)
            and will be space or tab seperated and that the bval file (the b values) are one one single line
            with space or tab separated b values.
        bval_scale (double or str): the amount by which we want to scale (multiply) the b-values.
            The default is auto, this checks if the first b-value is higher than 1e4 and if so multiplies it by
            1e-6 (sets bval_scale to 1e-6 and multiplies), else multiplies by 1.
    """
    b = protocol['b'].copy()
    g = protocol['g'].copy()

    if bval_scale == 'auto':
        if b[0] > 1e4:
            b *= 1e-6
    else:
        b *= bval_scale

    if column_based:
        b = b.transpose()
        g = g.transpose()

    for d in (bvec_fname, bval_fname):
        if not os.path.isdir(os.path.dirname(d)):
            os.makedirs(os.path.dirname(d))

    np.savetxt(bvec_fname, g)
    np.savetxt(bval_fname, b)


def load_protocol(data_source):
    """Load an protocol from the given protocol file, with as column names the given list of names.

    If the given file could not be found it tries once more by appending .prtcl to the end of the file.

    Args:
        data_source (string, Protocol): Either a filename, a directory path or a Protocol object to load.
            If a filename is given we load the protocol from file, if a directory is given we try to load a protocol
            from that directory. If an Protocol object is given we return it directly.

    Returns:
        :class:`Protocol`: An protocol object with all the columns loaded.
    """
    if isinstance(data_source, Protocol):
        return data_source

    if not os.path.isfile(data_source) and not os.path.isfile(data_source + '.prtcl'):
        if os.path.isdir(data_source):
            return auto_load_protocol(data_source)
        raise FileNotFoundError(data_source)

    if not os.path.isfile(data_source):
        data_source += '.prtcl'

    with open(data_source) as f:
        protocol = f.readlines()

    if protocol[0][0] != '#':
        raise ProtocolIOError('No column names defined in protocol.')

    column_names = [c.strip() for c in protocol[0][1:-1].split(',')]

    data = np.genfromtxt(data_source)
    s = data.shape
    d = {}

    if len(s) == 1:
        d.update({column_names[0]: data})
    else:
        for i in range(s[1]):
            d.update({column_names[i]: data[:, i]})

    return Protocol(columns=d)


def write_protocol(protocol, fname, columns_list=None):
    """Write the given protocol to a file.

    Args:
        protocol (Protocol): The protocol to write to file
        fname (string): The filename to write to
        columns_list (tuple): The tuple with the columns names to write (and in that order).
            If None, all the columns are written to file.

    Returns:
        tuple: the parameters that where written (and in that order)
    """
    if not columns_list:
        columns_list = protocol.column_names

        if 'G' in columns_list and 'Delta' in columns_list and 'delta' in columns_list:
            if 'maxG' in columns_list:
                columns_list.remove('maxG')

    data = protocol.get_columns(columns_list)

    if not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, 'w') as f:
        f.write('#')
        f.write(','.join(columns_list))
        f.write("\n")

    with open(fname, 'ab') as f:
        np.savetxt(f, data, delimiter="\t")

    if columns_list:
        return columns_list
    return protocol.column_names


def auto_load_protocol(directory, bvec_fname=None, bval_fname=None, bval_scale='auto', protocol_columns=None):
    """Load a protocol from the given directory.

    This function will only auto-search files in the top directory and not in the sub-directories.

    This will first try to use the first .prtcl file found. If none present, it will try to find bval and bvec files
    to use and then try to find the protocol options.

    The protocol_options should be a dictionary mapping protocol items to filenames. If given, we only use the items
    in that dictionary. If not given we try to autodetect the protocol option files from the given directory.

    The search order is (continue until matched):

        1) anything ending in .prtcl
        2) a) the given bvec and bval file
           b) anything containing bval or b-val
           c) anything containing bvec or b-vec
                i) This will prefer a bvec file that also has 'fsl' in the name. This to be able to auto use
                    HCP MGH bvec directions.
           d) protocol options
                i) using dict
                ii) matching filenames exactly to the available protocol options.
                    (e.g, finding a file named TE for the TE's)

    The available protocol options are:

        - TE: the TE in seconds, either a file or, one value or one value per bvec
        - TR: the TR in seconds, either a file or, either one value or one value per bvec
        - Delta: the big Delta in seconds, either a file or, either one value or one value per bvec
            can alternatively named ``big_delta``
        - delta: the small delta in seconds, either a file or, either one value or one value per bvec
        - maxG: the maximum gradient amplitude G in T/m. Used in estimating G, Delta and delta if not given.

    Args:
        directory (str): the directory to use the protocol from
        bvec_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)
        bval_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)
        bval_scale (double): The scale by which to scale the values in the bval file.
            If we use from bvec and bval we will use this scale. If 'auto' we try to guess the units/scale.
        protocol_columns (dict): mapping protocol items to filenames (as a subpath of the given directory)
            or mapping them to values (one value or one value per bvec line)

    Returns:
        Protocol: a loaded protocol file.

    Raises:
        ValueError: if not enough information could be found. (No protocol or no bvec/bval combo).
    """
    protocol_files = list(glob.glob(os.path.join(directory, '*.prtcl')))
    if protocol_files:
        return load_protocol(protocol_files[0])

    def filter_nifti_files(file_names):
        return list(filter(lambda s: not s.endswith('.nii') and not s.endswith('.nii.gz'), file_names))

    if not bval_fname:
        bval_files = filter_nifti_files(glob.glob(os.path.join(directory, '*bval*')))

        if not bval_files:
            bval_files = filter_nifti_files(glob.glob(os.path.join(directory, '*b-val*')))
            if not bval_files:
                raise ValueError('Could not find a suitable bval file')
        bval_fname = bval_files[0]

    if not bvec_fname:
        bvec_files = filter_nifti_files(glob.glob(os.path.join(directory, '*bvec*')))
        if not bvec_files:
            bvec_files = filter_nifti_files(glob.glob(os.path.join(directory, '*b-vec*')))
            if not bvec_files:
                raise ValueError('Could not find a suitable bvec file')

        for bvec_file in bvec_files:
            if 'fsl' in os.path.basename(bvec_file):
                bvec_fname = bvec_files[0]

        if not bvec_fname:
            bvec_fname = bvec_files[0]

    protocol = load_bvec_bval(bvec_fname, bval_fname, bval_scale=bval_scale)

    protocol_extra_cols = ['TE', 'TR', 'Delta', 'delta', 'maxG']

    if protocol_columns:
        for col in protocol_extra_cols:
            if col in protocol_columns:
                if isinstance(protocol_columns[col], str):
                    protocol = protocol.with_added_column_from_file(
                        col, os.path.join(directory, protocol_columns[col]))
                else:
                    protocol = protocol.with_new_column(col, protocol_columns[col])
    else:
        for col in protocol_extra_cols:
            if os.path.isfile(os.path.join(directory, col)):
                protocol = protocol.with_added_column_from_file(col, os.path.join(directory, col))

        # special case for Delta named as big_delta (Windows has a case-insensitive directory layout)
        if os.path.isfile(os.path.join(directory, 'big_delta')):
            protocol = protocol.with_added_column_from_file('Delta', os.path.join(directory, 'big_delta'))

    return protocol


def create_protocol(out_file=None, bvecs=None, bvals=None, **kwargs):
    """Create and write a protocol from the given keywords.

    Please note that all given columns should be in **SI units**.

    Args:
        out_file (str): the output filename, if not given we will not write the protocol.
        bvecs (str or ndarray): either an [n, 3] array or a string to a bvec file
        bvals (str or ndarray): either an [n, 1] array or a string to a bval file. This expects a typical bval file
            with units in s/mm^2.
        kwargs: other protocol columns, for example ``Delta=30e-3``

    Returns:
        Protocol: the created protocol
    """
    if bvecs is not None and bvals is not None:
        protocol = load_bvec_bval(bvecs, bvals)
    else:
        protocol = Protocol()

    if kwargs:
        protocol = protocol.with_updates(kwargs)

    if out_file:
        write_protocol(protocol, out_file)
    return protocol

