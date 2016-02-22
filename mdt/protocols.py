import glob
import logging
import numbers
import os
import itertools
import numpy as np
import copy
import six

__author__ = 'Robbert Harms'
__date__ = "2014-02-06"
__license__ = "LGPL v3"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


class Protocol(object):

    def __init__(self, columns=None):
        """Create a new protocol. Optionally initializes the protocol with the given set of columns.

        Args:
            columns (dict, optional, default None):
                The initial list of columns used by this protocol, the keys should be the name of the
                parameter (exactly as used in the model functions). The values should be numpy arrays of width 1, and
                all of equal length.
        """
        super(Protocol, self).__init__()
        self._gamma_h = 2.675987E8
        self._unweighted_threshold = 25e6
        self._columns = {}
        self._length = None
        self._logger = logging.getLogger(__name__)

        if columns:
            self._columns = columns
            self._length = columns[list(columns.keys())[0]].shape[0]

            for k, v in columns.items():
                s = v.shape
                if len(s) > 2 or (len(s) == 2 and s[1] > 1):
                    raise ValueError("All columns should be of width one.")

                if len(s) ==2 and s[1] == 1:
                    self._columns[k] = np.squeeze(v)

    @property
    def gamma_h(self):
        """Get the used gamma of the H atom used by this protocol.

        Returns:
            float: The used gamma of the H atom used by this protocol.
        """
        return self._gamma_h

    def add_column(self, name, data):
        """Add a column to this protocol. This overrides the column if present.

        Args:
            name (str): The name of the column to add
            data (ndarray): The vector to add to this protocol.

        Returns:
            self: for chaining
        """
        if isinstance(data, six.string_types):
            data = float(data)

        if isinstance(data, numbers.Number) or not data.shape:
            data = np.ones((self._length,)) * data

        s = data.shape
        if self._length and s[0] > self._length:
            self._logger.info("The column '{}' has to many elements ({}), we will only use the first {}.".format(
                name, s[0], self._length))
            self._columns.update({name: data[:self._length]})
        elif self._length and s[0] < self._length:
            raise ValueError("Incorrect column length given for '{}', expected {} and got {}.".format(
                name, self._length, s[0]))
        else:
            self._columns.update({name: data})

        return self

    def add_column_from_file(self, name, file_name, multiplication_factor=1):
        """Add a column to this protocol, loaded from the given file.

        The given file can either contain a single value (which is broadcasted), or one value per protocol line.

        Args:
            name (str): The name of the column to add
            file_name (str): The file to get the column from.
            multiplication_factor (double): we might need to scale the data by a constant. For example,
                if the data in the file is in ms we might need to scale it to seconds by multiplying with 1e-3
        Returns:
            self: for chaining
        """
        data = np.genfromtxt(file_name)
        data *= multiplication_factor
        self.add_column(name, data)
        return self

    def remove_column(self, column_name):
        """Completely remove a column from this protocol.

        Args:
            column_name (str): The name of the column to remove
        """
        if column_name == 'g':
            del self._columns['gx']
            del self._columns['gy']
            del self._columns['gz']
        else:
            if column_name in self._columns:
                del self._columns[column_name]

    def remove_rows(self, rows):
        """Remove a list of rows from all the columns.

        Please note that the protocol is 0 indexed.

        Args:
            rows (list of int): List with indices of the rows to remove
        """
        for key, column in self._columns.items():
            self._columns[key] = np.delete(column, rows)

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
        return self._length

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
        return list(self._columns.keys())

    def keys(self):
        return self._columns.keys()

    def get_nmr_shells(self):
        """Get the number of unique shells in this protocol.

        This is measured by counting the number of unique weighted bvals in this protocol.

        Returns:
            int: The number of unique weighted b-values in this protocol

        Raises:
            KeyError: This function may throw a key error if the 'b' column in the protocol could not be loaded.
        """
        return len(self.get_b_values_shells())

    def get_b_values_shells(self):
        """Get the b-values of the unique shells in this protocol.

        Returns:
            list: a list with the unique weighted bvals in this protocol.

        Raises:
            KeyError: This function may throw a key error if the 'b' column in the protocol could not be loaded.
        """
        return np.unique(self.get_column('b')[self.get_weighted_indices()]).tolist()

    def has_column(self, column_name):
        """Check if this protocol has a column with the given name.

        This will also return true if the column can be estimated from the other columns. See has_unestimated_column()
        to get information for columns that are really known.

        Returns:
            boolean: true if there is a column with the given name, false otherwise.
        """
        try:
            return self.get_column(column_name) is not None
        except KeyError:
            return False

    def has_unestimated_column(self, column_name):
        """Check if this protocol has real column information for the column with the given name.

        For example, the other function has_column('G') will normally return true since 'G' can be estimated from 'b'.
        This function however will return false if the column needs to be estimated and will
        return true if the column is truly known.

        Returns:
            boolean: true if there is really a column with the given name, false otherwise.
        """
        return column_name in self._columns

    def get_unweighted_indices(self, unweighted_threshold=None):
        """Get the indices to the unweighted volumes.

        Args:
            unweighted_threshold (float): the threshold under which we call it unweighted.

        Returns:
            list of int: A list of indices to the unweighted volumes.
        """
        unweighted_threshold = unweighted_threshold or self._unweighted_threshold

        b = self.get_column('b')
        g = self.get_column('g')

        g_limit = np.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2) < 0.99
        b_limit = b[:, 0] < unweighted_threshold

        return np.where(g_limit + b_limit)[0]

    def get_weighted_indices(self, unweighted_threshold=None):
        """Get the indices to the weighted volumes.

        Args:
            unweighted_threshold (float): the threshold under which we call it unweighted.

        Returns:
            list of int: A list of indices to the weighted volumes.
        """
        return sorted(set(range(self.get_column('b').shape[0])) -
                      set(self.get_unweighted_indices(unweighted_threshold=unweighted_threshold)))

    def get_indices_bval_in_range(self, start=0, end=1.0e9, epsilon=1e-5):
        """Get the indices of the b-values in the range [start - eps, end + eps].

        This can be used to get for example the indices of gradients whose b-value is in the range suitable for
        DTI analysis. To do so, use for example as start 0 and as end 1e9.

        Note that we use SI units and you need to specify the range in s/m^2 and not in s/mm^2.

        Also note that specifying 0 as start of the range does not automatically mean that the unweighted volumes are
        returned. Sometimes the b-value of the unweighted volumes is high while the gradient 'g' is [0 0 0]. This
        function does not make any assumptions about that and just returns indices in the given range, that gives:

        If you want to include the unweighted volumes, make a call to get_unweighted_indices() yourself.

        Args:
            start: b-value of the start of the range (inclusive) we want to get the indices of the volumes from.
                Should be positive. We subtract epsilon for float comparison
            end: b-value of the end of the range (inclusive) we want to get the indices of the volumes from.
                Should be positive. We add epsilon for float comparison
            epsilon: the epsilon we use in the range.

        Returns:
            list: a list of indices of all volumes whose b-value is in the given range.
                If you want to include the unweighted volumes, make a call to get_unweighted_indices() yourself.
        """
        b_values = self.get_column('b')
        return np.where(((start - epsilon) <= b_values) * (b_values <= (end + epsilon)))[0]

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

    def append_protocol(self, protocol):
        """Append another protocol to this protocol.

        This will add the columns of the other protocol to the columns of this protocol. This supposes both protocols
        have the same columns.
        """
        if type(protocol) is type(self):
            for key, value in self._columns.items():
                self._columns[key] = np.append(self[key], protocol[key], 0)

    def get_new_protocol_with_indices(self, indices):
        """Create a new protocol object with all the columns but as rows only those of the given indices.

        Args:
            indices: the indices we want to use in the new protocol

        Returns:
            Protocol: a protocol with all the data of the given indices
        """
        return Protocol(columns={k: v[indices] for k, v in self._columns.items()})

    def __len__(self):
        return self.length

    def __contains__(self, column):
        return self.has_column(column)

    def __getitem__(self, column):
        return self.get_column(column)

    def __str__(self):
        s = 'Column names: ' + ', '.join(self.column_names) + "\n"
        s += 'Data: ' + "\n"
        s += np.array_str(self.get_all_columns())
        return s

    def _get_real_column(self, column_name):
        """Try to load a real column from this protocol.

        Returns:
            A real column, that is, a column from which we have real data.

        Raises:
            KeyError: If the column name could not be found we raise a key error.
        """
        if column_name in self._columns:
            return np.reshape(self._columns[column_name], (-1, 1))

        if column_name == 'g':
            return self.get_columns(('gx', 'gy', 'gz'))

        raise KeyError('The given column could not be found.')

    def _get_estimated_column(self, column_name):
        """Try to load an estimated column from this protocol.

        Returns:
            An estimated column, that is, a column we estimate from the other columns.

        Raises:
            KeyError: If the column name could not be estimated we raise a key error.
        """
        if column_name == 'maxG':
            return np.ones((self._length,)) * 0.04

        sequence_timings = self._get_sequence_timings()

        if column_name in sequence_timings:
            return sequence_timings[column_name]

        if column_name == 'q':
            return np.reshape((self._gamma_h * sequence_timings['G'] * sequence_timings['delta'] / (2 * np.pi)),
                              (-1, 1))

        if column_name == 'GAMMA2_G2_delta2':
            return np.reshape(np.power(self._gamma_h * sequence_timings['G'] * sequence_timings['delta'], 2), (-1, 1))

        if column_name == 'b':
            return np.reshape(self._gamma_h ** 2 *
                              sequence_timings['G']**2 *
                              sequence_timings['delta']**2 *
                              (sequence_timings['Delta'] - (sequence_timings['delta']/3)), (-1, 1))

        raise KeyError('The given column name "{}" could not be found in this protocol.'.format(column_name))

    def _get_sequence_timings(self):
        """Return G, Delta and delta, estimate them if necessary.

        If Delta and delta are available, they are used instead of estimated Delta and delta.

        Returns:
            the columns G, Delta and delta
        """
        if all(map(lambda v: v in self._columns, ['b', 'Delta', 'delta'])):
            G = np.sqrt(self._columns['b'] / (self.gamma_h**2 * self._columns['delta']**2 *
                                             (self._columns['Delta'] - (self._columns['delta']/3.0))))
            G[self.get_unweighted_indices()] = 0
            return {'G': G, 'Delta': self._columns['Delta'], 'delta': self._columns['delta']}

        if all(map(lambda v: v in self._columns, ['b', 'Delta', 'G'])):
            roots = np.roots([-1/3.0, self._columns['Delta'],
                              -self._columns['b']/(self._gamma_h**2 * self._columns['G']**2)])
            delta = list(itertools.dropwhile(np.isreal, roots))[0]
            return {'G': self._columns['G'], 'Delta': self._columns['Delta'], 'delta': delta}

        if all(map(lambda v: v in self._columns, ['b', 'G', 'delta'])):
            Delta = ((self._columns['b'] - self._gamma_h**2 * self._columns['G']**2 * self._columns['delta']**3/3.0) /
                        (self._gamma_h**2 * self._columns['G']**2 * self._columns['delta']**2))
            return {'G': self._columns['G'], 'delta': self._columns['delta'], 'Delta': Delta}

        if all(map(lambda v: v in self._columns, ['G', 'delta', 'Delta'])):
            return {'G': self._columns['G'], 'delta': self._columns['delta'], 'Delta': self._columns['Delta']}

        if 'b' not in self._columns:
            return {}

        maxG = self.get_column('maxG')
        bvals = self.get_column('b')
        bmax = max(self.get_b_values_shells())

        Deltas = (3 * bmax / (2 * self.gamma_h**2 * maxG**2))**(1 / 3.0)
        deltas = Deltas
        G = np.sqrt(bvals / bmax) * maxG

        return {'G': G, 'Delta': Deltas, 'delta': deltas}


def load_bvec_bval(bvec_file, bval_file, column_based='auto', bval_scale='auto'):
    """Load an BG scheme object from a bvec and bval file.

    If column_based
    This supposes that the bvec (the vector file) has 3 rows (gx, gy, gz) and is space or tab seperated.
    The bval file (the b values) are one one single line with space or tab separated b values.

    Args:
        bvec_file (str): The filename of the bvec file
        bval_file (str): The filename of the bval file
        column_based (boolean): If true, this supposes that the bvec (the vector file) has 3 rows (gx, gy, gz)
            and is space or tab seperated and that the bval file (the b values) are one one single line
            with space or tab separated b values.
            If false, the vectors and b values are each one a different line.
            If 'auto' it is autodetected, this is the default.
        bval_scale (float): The amount by which we want to scale (multiply) the b-values. The default is auto,
            this checks if the b-val is lower then 1e4 and if so multiplies it by 1e6.
            (sets bval_scale to 1e6 and multiplies), else multiplies by 1.

    Returns:
        Protocol the loaded protocol.
    """
    bvec = np.genfromtxt(bvec_file)
    bval = np.expand_dims(np.genfromtxt(bval_file), axis=1)

    if bval_scale == 'auto' and bval[0, 0] < 1e4:
        bval *= 1e6
    else:
        bval *= bval_scale

    if len(bvec.shape) < 2:
        raise ValueError('Bval file does not have enough dimensions.')

    if column_based == 'auto':
        if bvec.shape[1] > bvec.shape[0]:
            bvec = bvec.transpose()
    elif column_based:
        bvec = bvec.transpose()

    columns = {'gx': bvec[:, 0], 'gy': bvec[:, 1], 'gz': bvec[:, 2], 'b': bval}

    if bvec.shape[0] != bval.shape[0]:
        raise ValueError('Columns not of same length.')

    return Protocol(columns=columns)


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


def load_protocol(protocol_fname, column_names=None):
    """Load an protocol from the given protocol file, with as column names the given list of names.

    Args:
        protocol_fname (string): The filename of the protocol file to load.
            This should be a comma seperated, or tab delimited file with equal length columns.
        column_names (tuple): A tuple or list of the columns names. Please note that every column should be named.
            The gradient vector for example should be listed as 'gx', 'gy', 'gz'.

    Returns:
        An protocol with all the columns loaded.
    """
    with open(protocol_fname) as f:
        protocol = f.readlines()

    if not column_names:
        if protocol[0][0] == '#':
            line = protocol[0][1:-1]
            sep = ' '
            if ',' in line:
                sep = ','
            cols = line.split(sep)
            cols = [c.strip() for c in cols]
            column_names = cols
        else:
            ValueError('No column names given and none in protocol file.')

    data = np.genfromtxt(protocol_fname)
    s = data.shape
    d = {}

    if len(s) == 1:
        d.update({column_names[0]: data})
    else:
        for i in range(s[1]):
            d.update({column_names[i]: data[:, i]})
    return Protocol(columns=d)


def column_names_nice_ordering(column_names, preferred_order=None):
    """Order the column names to a nice preferred order.

    Args:
        column_names (list of str): the list with column names
        preferred_order (list of str): the preferred partial ordering

    Returns:
        the same list of column names ordered to the given partial ordering
    """
    columns_list = list(reversed(column_names))
    preferred_order = preferred_order or ('gx', 'gy', 'gz', 'G', 'Delta', 'delta', 'TE', 'T1', 'b', 'q', 'maxG')

    final_list = []
    for p in preferred_order:
        if p in columns_list:
            columns_list.remove(p)
            final_list.append(p)
    final_list.extend(columns_list)

    return final_list


def write_protocol(protocol, fname, columns_list=None):
    """Write the given protocol to a file.

    This writes all or the selected columns from the given protocol to the given file name.

    Args:
        protocol (Protocol): The protocol to write to file
        fname (string): The filename to write to
        columns_list (tuple, optional, default None): The tuple with the columns names to write (and in that order).
            If None, all the columns are written to file.

    Returns:
        A tuple listing the parameters that where written (and in that order)
    """
    if not columns_list:
        columns_list = column_names_nice_ordering(protocol.column_names)

        if 'G' in columns_list and 'Delta' in columns_list and 'delta' in columns_list:
            if 'b' in columns_list:
                columns_list.remove('b')
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


def auto_load_protocol(directory, protocol_options=None, bvec_fname=None, bval_fname=None, bval_scale='auto'):
    """Load a protocol from the given directory.

    This function will only auto-search files in the top directory and not in the sub-directories.

    This will first try to load the first .prtcl file found. If none present, it will try to find bval and bvec files
    to load and then try to find the protocol options.

    The protocol_options should be a dictionary mapping protocol items to filenames. If given, we only use the items
    in that dictionary. If not given we try to autodetect the protocol option files from the given directory.

    The search order is (continue until matched):
        1) anything ending in .prtcl
        2) a) the given bvec and bval file
           b) anything containing bval or b-val
           c) anything containing bvec or b-vec
           d) protocol options
                i) using dict
                ii) matching filenames exactly to the available protocol options.
                    (e.g, finding a file named TE for the TE's)

    The available protocol options are:
        - TE: the TE in seconds, either a file or, one value or one value per bvec
        - TR: the TR in seconds, either a file or, either one value or one value per bvec
        - Delta: the big Delta in seconds, either a file or, either one value or one value per bvec
        - delta: the small delta in seconds, either a file or, either one value or one value per bvec
        - maxG: the maximum gradient amplitude G in T/m. Used in estimating G, Delta and delta if not given.

    Args:
        directory (str): the directory to load the protocol from
        protocol_options (dict): mapping protocol items to filenames (as a subpath of the given directory)
            or mapping them to values (one value or one value per bvec line)
        bvec_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)
        bval_fname (str): if given, the filename of the bvec file (as a subpath of the given directory)
        bval_scale (double): The scale by which to scale the values in the bval file.
            If we load from bvec and bval we will use this scale. If 'auto' we try to guess the units/scale.

    Returns:
        Protocol: a loaded protocol file.

    Raises:
        ValueError: if not enough information could be found. (No protocol or no bvec/bval combo).
    """
    protocol_files = list(glob.glob(os.path.join(directory, '*.prtcl')))
    if protocol_files:
        return load_protocol(protocol_files[0])

    if not bval_fname:
        bval_files = list(glob.glob(os.path.join(directory, '*bval*')))
        if not bval_files:
            bval_files = glob.glob(os.path.join(directory, '*b-val*'))
            if not bval_files:
                raise ValueError('Could not find a suitable bval file')
        bval_fname = bval_files[0]

    if not bvec_fname:
        bvec_files = list(glob.glob(os.path.join(directory, '*bvec*')))
        if not bvec_files:
            bvec_files = glob.glob(os.path.join(directory, '*b-vec*'))
            if not bvec_files:
                raise ValueError('Could not find a suitable bvec file')
        bvec_fname = bvec_files[0]

    protocol = load_bvec_bval(bvec_fname, bval_fname, bval_scale=bval_scale)

    protocol_extra_cols = ['TE', 'TR', 'Delta', 'delta', 'maxG']

    if protocol_options:
        for col in protocol_extra_cols:
            if col in protocol_options:
                if isinstance(protocol_options[col], six.string_types):
                    protocol.add_column_from_file(col, os.path.join(directory, protocol_options[col]))
                else:
                    protocol.add_column(col, protocol_options[col])
    else:
        for col in protocol_extra_cols:
            if os.path.isfile(os.path.join(directory, col)):
                protocol.add_column_from_file(col, os.path.join(directory, col))

    return protocol
