import numbers
import os
import numpy as np
import copy

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

        Attributes:
            max_G (double): The maximum G value in T/m. Used in estimating G, Delta and delta if not given.
        """
        super(Protocol, self).__init__()
        self.max_G = 0.04
        self._gamma_h = 2.675987E8
        self._unweighted_threshold = 25e6
        self._columns = {}
        self._length = None

        if columns:
            self._columns = columns
            self._length = columns[list(columns.keys())[0]].shape[0]

            for v in columns.values():
                s = v.shape
                if len(s) > 2 or (len(s) == 2 and s[1] > 1):
                    raise ValueError("All columns should be of width one.")

    @property
    def gamma_h(self):
        """Get the used gamma of the H atom used by this protocol.

        Returns:
            float: The used gamma of the H atom used by this protocol.
        """
        return self._gamma_h

    def add_column(self, name, data):
        """Add a column to this protocol.

        Args:
            name (str): The name of the column to add
            data (ndarray): The vector to add to this protocol.

        Returns:
            self: for chaining
        """
        if isinstance(data, numbers.Number):
            data = np.ones((self._length,)) * data
        elif not data.shape:
            data = np.ones((self._length,)) * data

        s = data.shape
        if self._length and s[0] != self._length:
            raise ValueError("Incorrect column length given.")
        self._columns.update({name: data})

        if name == 'delta':
            if 'Delta' in self._columns and 'G' not in self._columns:
                self.add_column('G', self._estimate_sequence_timings()['G'])
        elif name == 'Delta':
            if 'delta' in self._columns and 'G' not in self._columns:
                self.add_column('G', self._estimate_sequence_timings()['G'])

        return self

    def add_column_from_file(self, name, file_name):
        """Add a column to this protocol, loaded from the given file.

        The given file can either contain a single value (which is broadcasted), or one value per protocol line.

        Args:
            name (str): The name of the column to add
            data (ndarray): The file to get the column from.

        Returns:
            self: for chaining
        """
        data = np.genfromtxt(file_name)
        self.add_column(name, data)
        return self

    def remove_column(self, column_name):
        """Completely remove a column from this protocol.

        Args:
            name (str): The name of the column to remove
        """
        if column_name == 'g':
            del self._columns['gx']
            del self._columns['gy']
            del self._columns['gz']
        else:
            del self._columns[column_name]

    def remove_rows(self, rows):
        """Remove a list of rows from all the columns.

        Args:
            rows (list of int): List with indices of the rows to remove
        """
        for key, column in self._columns.items():
            self._columns[key] = np.delete(column, rows)

    def get_column(self, column_name):
        """Get the column associated by the given column name.

        Args:
            column_name (str): The name of the column we want to return.

        Returns:
            ndarray: The column we would like to return. This is returned as a 2d matrix with shape (n, 1).

        Raises:
            KeyError: If the column could not be found.
        """
        if column_name in self._columns:
            return np.reshape(self._columns[column_name], (-1, 1))

        if column_name == 'g':
            return self.get_columns(('gx', 'gy', 'gz'))

        if column_name == 'q':
            if self.has_column('G') and self.has_column('delta'):
                return np.reshape(self._gamma_h * self.get_column('G') *
                                  self.get_column('delta') / (2 * np.pi), (-1, 1))

        if column_name == 'GAMMA2_G2_delta2':
            if self.has_column('G') and self.has_column('delta'):
                return np.reshape(np.power(self._gamma_h * self.get_column('G') * self.get_column('delta'), 2), (-1, 1))

        if column_name == 'b':
            if self.has_column('G') and self.has_column('delta') and self.has_column('Delta'):
                return np.reshape(self._gamma_h ** 2 *
                                  self.get_column('G')**2 *
                                  self.get_column('delta')**2 *
                                  (self.get_column('Delta') - (self.get_column('delta')/3)), (-1, 1))

        if column_name == 'G':
            return self._estimate_sequence_timings()['G']

        if column_name == 'Delta':
            return self._estimate_sequence_timings()['Delta']

        if column_name == 'delta':
            return self._estimate_sequence_timings()['delta']

        raise KeyError('The given column name "{}" could not be found in this protocol.'.format(column_name))

    @property
    def protocol_length(self):
        """Get the length of this protocol.

        Returns:
            int: The length of the protocol.
        """
        return self._length

    @property
    def number_of_columns(self):
        """Get the number of columns in this protocol.

        Returns:
            int: The number columns in this protocol.
        """
        return len(self._columns)

    @property
    def column_names(self):
        """Get the names of the columns.

        Returns:
            list of str: The names of the columns.
        """
        return self._columns.keys()

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
        return np.unique(self.get_column('b')[self.get_weighted_indices()])

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

        For example, has_column('G') will always return true since 'G' can be estimated from 'b'. This function
        however will return false if the column needs to be estimated and will return true if the column is truly known.

        Returns:
            boolean: true if there is really a column with the given name, false otherwise.
        """
        return column_name in self._columns

    def get_columns(self, column_names):
        """Get a matrix containing the requested column names in the order given.

        Returns:
            ndarrray: A 2d matrix with the column requested concatenated.
        """
        if not column_names:
            return None
        return np.concatenate([self[i] for i in column_names], axis=1)

    def get_unweighted_indices(self):
        """Get the indices to the unweighted volumes.

        Returns:
            list of int: A list of indices to the unweighted volumes.
        """
        b = self.get_column('b')
        g = self.get_column('g')

        g_limit = np.sqrt(g[:, 0]**2 + g[:, 1]**2 + g[:, 2]**2) < 0.99
        b_limit = b[:, 0] < self._unweighted_threshold

        return np.where(g_limit + b_limit)[0]

    def get_weighted_indices(self):
        """Get the indices to the weighted volumes.

        Returns:
            list of int: A lsit of indices to the weighted volumes.
        """
        return sorted(set(range(self.get_column('b').shape[0])) - set(self.get_unweighted_indices()))

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
        """Get all columns as a big array.

        Returns:
            ndarray: All the columns of this protocol.
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
        return self.protocol_length

    def __contains__(self, column):
        return self.has_column(column)

    def __getitem__(self, column):
        return self.get_column(column)

    def __str__(self):
        s = 'Column names: ' + ', '.join(self.column_names) + "\n"
        s += 'Data: ' + "\n"
        s += np.array_str(self.get_all_columns())
        return s

    def add_estimated_protocol_params(self, maxG=None, Delta=None, delta=None):
        maxG = maxG or self.max_G
        if Delta is not None:
            self.add_column('Delta', Delta)
        if delta is not None:
            self.add_column('delta', delta)

        items = self._estimate_sequence_timings(max_G=maxG)
        if Delta is None:
            self.add_column('Delta', items['Delta'])
        if delta is None:
            self.add_column('delta', items['delta'])
        self.add_column('G', items['G'])

    def _estimate_sequence_timings(self, max_G=None):
        """Return estimated G, Delta and delta.

        If Delta and delta are available, they are used instead of estimated Delta and delta.

        Args:
            maxG (double): The maximum G value in T/m
            Delta (double): If set, use this Delta
            delta (double): If set, use this delta

        Returns:
            the columns G, Delta and delta
        """
        if 'b' in self._columns and 'Delta' in self._columns and 'delta' in self._columns:
            G = np.sqrt(self.get_column('b') / (self.gamma_h**2 * self.get_column('delta')**2 *
                                                (self.get_column('Delta') - (self.get_column('delta')/3.0))))
            G[self.get_unweighted_indices()] = 0
            return {'G': G, 'Delta': self._columns['Delta'], 'delta': self._columns['delta']}

        max_G = max_G or self.max_G
        bvals = self.get_column('b')
        bmax = max(self.get_b_values_shells())

        Deltas = np.ones_like(bvals) * (3 * bmax / (2 * self.gamma_h**2 * max_G**2))**(1/3.0)
        deltas = Deltas
        G = np.sqrt(bvals / bmax) * max_G

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
    b = protocol['b']
    g = protocol['g']

    if bval_scale == 'auto' and b[0] > 1e4:
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

    data = np.genfromtxt(protocol)
    s = data.shape
    d = {}
    for i in range(s[1]):
        d.update({column_names[i]: data[:, i]})
    return Protocol(columns=d)


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
        columns_list = list(reversed(protocol.column_names))
        preferred_order = ('gx', 'gy', 'gz', 'G', 'Delta', 'delta', 'TE', 'T1', 'b', 'q')

        if 'G' in columns_list and 'Delta' in columns_list and 'delta' in columns_list:
            if 'b' in columns_list:
                columns_list.remove('b')

        final_list = []
        for p in preferred_order:
            if p in columns_list:
                columns_list.remove(p)
                final_list.append(p)
        final_list.extend(columns_list)
        columns_list = final_list

    data = protocol.get_columns(columns_list)

    if not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, 'w') as f:
        f.write('#')
        f.write(','.join(columns_list))
        f.write("\n")
        np.savetxt(f, data, delimiter="\t")

    if columns_list:
        return columns_list
    return protocol.column_names
