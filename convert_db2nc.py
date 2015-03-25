#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert database structured txt-files into netCDF format.

For usage from command line see
    python convert_db2nc.py -h

Requirements:
    - numpy
    - netCDF4
    - argparse

Converter class: DB2NC

Check out current version at
https://github.com/Chilipp/Convert-Database-to-NetCDF
"""
import os
import pickle
from collections import OrderedDict
from argparse import ArgumentParser
from netCDF4 import Dataset
from numpy import transpose, loadtxt, unique, shape, zeros, array, roll, \
    dstack, ones, set_printoptions, get_printoptions
from numpy import all as npall
from numpy import any as npany
from numpy.ma import masked_where, filled
from datetime import datetime, timedelta
from itertools import izip, chain, product, imap
import sys

__author__ = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"
__version__ = "1.0"


missval = -9999

# dictionary containing documentation for keyword arguments
# (used for the argument parser and in the methods of DB2NC class)
kwargdocs = {'init': OrderedDict((  # init specific options
    ('ifile', """
        Path to the text file containing the information in columns which
        shall be converted into NetCDF."""),
    ('gridcols', """
        List of integers representing the columns with the
        grid info of what shall be considered in inputfile. The sorting and
        number of columns must correspond to the variables for the
        mask-file."""),
    ('defaultflags', """
        List of integers representing the columns with the
        flags of what shall be considered in inputfile and do not belong to
        concatenation columns (see cat option)."""),
    ('mask', """
        Netcdf file containing grid definition masks stored
        in the specified variable names, separated by comma. The sorting and
        number of variables must correspond to the gridcols."""),
    ('alias', """
        Use aliases for the items in col (for gridcols only) as specified in
        file. The given file must contain two columns: the first one consists
        of the unique items from ifile, the second one of their aliases."""),
    ('noheader', """
        If set, first rows of all text files is not omitted"""),
    ('cat', """
        Create variables in the NetCDF file concatenated from the given
        column pairs. For each argument col1,col2 the variables in the
        NetCDF-file will be like flag1_flag2 where flag1 is out of the items
        of the col1-th column in inputfile and flag2 out of the col2-th column
        in inputfile"""),
    ('sort', """
        Sort items in column col as specified in file. File must contain two
        columns: the first columns consists the unique flag names of the
        inputfile, the second column consists of the final names for which to
        sort (see also option -redistribute)."""),
    ('redistribute', """
        Sort item flag in column col as specified in file by its fraction in
        each gridcell. File must contain 2-dimensional fields stored in the
        given varnames representing the fraction of the flag in the given
        itemname for each gridcell."""),
    ('valcol', """
        Column containing the value in inputfile which will be added up.
        Default: last, i.e. -1"""),
    ('time', """
        Columns with time information and format, separated by comma.""")
    ))}
kwargdocs['convert'] = OrderedDict((  # convert specific options
    ('verbose', """
        If set, use verbose"""),
    ('maxrows', """
        Number of rows to loop through in ifiledata""")
    ))
kwargdocs['output'] = OrderedDict((
    ('output', """
        Output filename."""),
    ('header', """
     Dictionary containing header information (e.g. {'title': 'test'})"""),
    ('metadata', """
     Dictionary containing metadata information. Keys are flagname, values are
     dictionaries with items as (attribute, value). One example for variable
     'tair': {'tair': {'long_name': 'Air Temperature',
                       'units': 'K'}}"""),
    ('clobber', """
     Enable clobber (will significantly reduce file size). Input must be 'auto'
     or a list of the chunking parameters (the first one corresponds to time,
     the others to the dimension as stored in the netCDF file (usually the
     second corresponds to lat, the third to lon).
     If 'auto' chunking parameters are deterimined such that 1D and 2D access
     are balanced. The calculation function is taken from
         http://www.unidata.ucar.edu/staff/russ/public/chunk_shape_3D.py"""),
    ('compression', """
     Dictionary with compression parameters for netCDF4 variable (determined by
     netCDF4 package. Possible keywords are zlib, complevel, shuffle and
     least_significant_digit. For documentation see

http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4.Variable-class.html
     If compression is not a dictionary, the value will be used for the zlib
     keyword in netCDF4 variables.""")
    ))


def loadpickle(string):
    """Function to load a dictionary with pickle.load(f), where f is the file
    handler of the 'string')"""
    with open(string) as f:
        val = pickle.load(f)
    return val


def determine_chunk(string):
    if string == 'auto':
        return string
    else:
        return map(int, string.split(','))


class MyParser(ArgumentParser):
    """Subclass of ArgumentParser modified convert_arg_line_to_args method
    to read in multiple arguments from one line in an input file and to
    enable commands."""
    def convert_arg_line_to_args(self, arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg
    convert_arg_line_to_args.__doc__ = \
        ArgumentParser.convert_arg_line_to_args.__doc__

parser = MyParser(
    description="%s\nAuthor: %s\nVersion: %s" % (__doc__, __author__,
                                                 __version__),
    usage='%(prog)s [options]',
    fromfile_prefix_chars="@")
parser.add_argument('--version', action='version', version=__version__)
parser.add_argument(
    'ifile', help=kwargdocs['init']['ifile'],
    metavar='<<<inputfile>>>')
parser.add_argument(
    'mask',
    help=kwargdocs['init']['mask'],
    metavar='<<<mask-file>>>,<<<var1>>>,<<<var2>>>,...')
parser.add_argument('-v', '--verbose', help=kwargdocs['convert']['verbose'],
                      action='store_true')
parser.add_argument(
    '-info',
    help="""
    Same as verbose option (-v) but only makes the initialization, print
    information on the final output and exit.""",
    action='store_true')
parser.add_argument(
    '-noheader',
    help=kwargdocs['init']['noheader'],
    action='store_true')

# spatial reference columns
_colgrp = parser.add_argument_group(
    'Column numbers',
    """Set up how the column shall be used.""")
_colgrp.add_argument(
    '-gridcols',
    help=kwargdocs['init']['gridcols'],
    metavar='<<<gridcol>>>', required=True, type=int, nargs='*')
# flag columns
_colgrp.add_argument(
    '-flagcols',
    help=kwargdocs['init']['defaultflags'],
    metavar='<<<flagcol>>>', type=int, default=[], nargs='*',
    dest='defaultflags')
_colgrp.add_argument(
    '-cat',
    help=kwargdocs['init']['cat'],
    nargs='*', dest='cat', metavar='<<<col1>>>,<<<col2>>>', default=[])
# value column
_colgrp.add_argument(
    '-valcol',
    help='Default: %(default)s. ' + kwargdocs['init']['valcol'],
    default=-1, type=int)
# time column
_colgrp.add_argument(
    '-t', '--time',
    help=kwargdocs['init']['time'],
    metavar="<<<col>>>,<<<format>>>", nargs='*', default=None)

_colhandlegrp = parser.add_argument_group(
    'Special treatment of columns',
    """Set up if columns shall be treated in a special manner.""")
_colhandlegrp.add_argument(
    '-alias',
    help=kwargdocs['init']['alias'],
    metavar='<<<col>>>,<<<file>>>', nargs='*', default=[])
_colhandlegrp.add_argument(
    '-sort',
    help=kwargdocs['init']['sort'],
    metavar='<<<col>>>,<<<file>>>', nargs='*', default=[])
_colhandlegrp.add_argument(
    '-rd', '--redistribute',
    help=kwargdocs['init']['redistribute'],
    metavar=(
        '<<<col>>>,<<<flag>>>,<<<file>>>,<<<item1>>>,<<<var1>>>'
        '[,<<<item2>>>,<< <var2>>>,...]'),
    nargs='*', default=[])
# misc
_miscgrp = parser.add_argument_group('Miscellaneous Output Options')
_miscgrp.add_argument(
    '-o', '--output',
    help=' Default: %(default)s. ' + kwargdocs['output']['output'],
    default='landuse.nc', metavar='<<<file name>>>')
_miscgrp.add_argument(
    '-c', '--clobber',
    help="Default: %(default)s, without arguments: %(const)s. " +
    kwargdocs['output']['clobber'],
    nargs='?', default=False, const='auto',
    type=determine_chunk,
    metavar='chunk_time,chunk_lon,chunk_lat')
_miscgrp.add_argument(
    '-z', '--compression', nargs='?', default=False, const=True,
    type=lambda x: {'zlib': True, 'complevel': int(x)},
    help="""
    Enable compression. Without arguments, simply set zlib=True in netCDF
    variables (i.e. compression level is 4).
    Arguments may be integer between 1 and 9 determining the compression
    level.""",
    metavar='<<<complevel>>>')
_miscgrp.add_argument(
    '-meta', '--metadata', default={},
    help="""
    Pickle file (i.e. dictionary stored as pickle file) containing metadata
    information. Keys are flagname, values are dictionaries with items as
    (attribute, value). One example for variable 'tair':
        {'tair': {'long_name': 'Air Temperature',
                  'units': 'K'}}""",
    type=loadpickle,
    metavar='<<<pickle file>>>')
_miscgrp.add_argument(
    '-header', default={},
    help="""
    Pickle file (i.e. dictionary stored as pickle file) containing header
    information for NetCDF file. For example: {'title': 'test'}""",
    type=loadpickle,
    metavar='<<<pickle file>>>')
_miscgrp.add_argument(
    '-rows', '--maxrows', help=kwargdocs['convert']['maxrows'], type=int)


def main(*args):
    t = datetime.now()
    if args != ():
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()
    opt, initkwargs, loopkwargs, outputkwargs = set_options(*args)
    converter = DB2NC(**initkwargs)
    if opt.info:  # if test option enabled, quit
        converter.info()
        return converter
    elif opt.verbose:
        converter.info()
    converter.convert(**loopkwargs)
    converter.output_nc(**outputkwargs)
    print(datetime.now()-t)
    return converter


def set_options(*args):
    """Function to set up the keyword arguments for the methods of DB2NC class
    from the Argument Parser.

    Arguments will be passed to the parser.

    Return:
      - initkwargs: Dictionary with keyword arguments for __init__ method of
          DB2NC instance
      - loopkwargs: Dictionary with keyword arguments for convert method of
          DB2NC instance
      - outputkwargs: Dictionary with keyword arguments for output_nc method
          of DB2NC instance
    """
    if args != ():
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()
    initkwargs = {key: val for key, val in vars(opt).items()
                  if key in kwargdocs['init']}
    loopkwargs = {key: val for key, val in vars(opt).items()
                  if key in kwargdocs['convert']}
    outputkwargs = {key: val for key, val in vars(opt).items()
                    if key in kwargdocs['output']}
    return opt, initkwargs, loopkwargs, outputkwargs


class Redistributer(object):
    """class which stores in data attribute how to redistribute the data"""
    __slots__ = ('col', 'flag', 'data', 'ncfile', 'variables')

    def __init__(self, sortitem):
        splitteditem = sortitem.split(',')
        # col attribute: Column in ifile which includes the flag to
        # redistribute
        self.col = int(splitteditem[0])
        # flag attribute: Name of the flag which shall be redistributed
        self.flag = splitteditem[1]
        self.ncfile = splitteditem[2]  # NetCDF file name
        self.variables = {  # variables and their counterpart in ncfile
            aliasflag: var for aliasflag, var in
            izip(splitteditem[3::2], splitteditem[4::2])}
        # --- read redistribution data
        nco = Dataset(splitteditem[2])
        # data attribute: Dictionary with target flag as keys and
        # fractions for the redistribution as values
        self.data = {
            aliasflag: nco.variables[var][0, :] for aliasflag, var in
            izip(splitteditem[3::2], splitteditem[4::2])}
        nco.close()

    def info(self):
        print('Redistribute %s in column %i with variables from %s' % (
            self.flag, self.col, self.ncfile))
        print('   Variables in NetCDF file correspond to %s' % (
            ', '.join('%s (%s)' % item for item in self.variables.items())))


class Adder(object):
    """Class determining how the data shall be added and to where"""

    def __init__(self, target, mulc=None):
        """
        Input:
            - target: Array where the data of addfunc will be added
            - mulc: multiplication number (if given, use redist method (for
            redistribution data), else use addnormal)"""
        self.target = target
        if mulc is not None:
            self.mulc = mulc
            self.addfunc = self.redist
        else:
            self.addfunc = self.addnormal

    def redist(self, itime, fields, val):
        """method for redistribution"""
        self.target[itime, :][fields] += val*self.mulc[fields]

    def addnormal(self, itime, fields, val):
        """method for normal add"""
        self.target[itime, :][fields] += val


class LoopInfo(object):
    """Class giving information about the loop"""
    __slots__ = ('total', 'counter', 't0', 'info')

    def __init__(self, total, t0, verbose):
        """
        Input:
          - total: Integer. Total lenght of loop
          - t0: datetime object of initial time
          - verbose: If True, will print update to stdout, else do nothing
        """
        self.counter = 0
        self.total = total
        self.t0 = t0
        if verbose:
            self.info = self._verboseprint
        else:
            self.info = self._donothing

    def _donothing(self):
        pass

    def _verboseprint(self):
        """Function which will be called if verbose is set at initialization"""
        if not self.counter % 100:
            if self.counter == 0:
                sys.stdout.write(
                    "\rProcessed %i of %i." % (self.counter, self.total))
                sys.stdout.flush()
            else:
                sys.stdout.write(
                    ("\rProcessed %i of %i. "
                     "Remaining time: %4.1f minutes.") % (
                         self.counter, self.total,
                         # expected time
                         (datetime.now() - self.t0).total_seconds() *
                         self.total / (60. * self.counter) -
                         # time already passed
                         (datetime.now() - self.t0).total_seconds()/60.
                    ))
                sys.stdout.flush()
        self.counter += 1


class Data(object):
    # container for large data arrays
    __slots__ = ['ifiledata', 'finaldata', 'maskdata', 'lat', 'lon']

    def __init__(self):
        pass


class DB2NC(object):
    """Class to convert database structured data to netCDF file"""

    def __init__(self, ifile, mask, gridcols, defaultflags=[], alias=[],
                 noheader=False, cat=[], sort=[], redistribute=[],
                 weights=None, valcol=-1, time=None):
        """Initialization function for DB2NC class"""
        # docstring is extended below
        self.data = Data()  # data container
        if not noheader:
            kwargs = {'skiprows': 1}
        else:
            kwargs = {}

        # set up column data
        self._set_cols(gridcols=gridcols, time=time,
                       defaultflags=defaultflags, cat=cat,
                       valcol=valcol)

        # read data from file
        self._read_data(ifile=ifile, **kwargs)

        # read grid data
        self._read_grid(mask=mask)

        # read data for columns with alias
        self.read_alias(alias=alias)

        # set up time data
        self._set_up_timedata()

        # --- handling flags ---
        self._read_sorting(sort, **kwargs)
        self._read_redist(redistribute)

        # set up variable names and initialize data
        self._set_names()

        # set up how data shall be added to data array
        self._set_add()

    def info(self):
        """Print information about the DB2NC instance"""
        # store current numpy print options
        printops = get_printoptions()
        # limit size of printed numpy arrays
        set_printoptions(threshold=5, edgeitems=3)
        # input file name
        print('Input file: ' + self.ifile)
        # mask file name
        print('Mask file with grid informations: ' + self.maskfile)
        # value column
        print('Column containing value: %i' % self.valcol)
        # grid columns
        print('Columns containing spatial information: %s' % (
            ', '.join(map(str, self.gridcols))))
        # flag columns
        print('Columns with flag definitions: %s' % (
            ', '.join(map(str, self.flagcols))))
        # concatenation columns
        print('Columns that shall be concatenated: %s' % (
            ', '.join(map(lambda x: ' and '.join(imap(str, x)),
                          self.catcolpairs))))
        # time column
        print('Columns with time information: ' + (
            ', '.join(map(str, self.timecols))))
        # number of time steps
        print('Number of timesteps found: %i' % self.ntimes)
        # time information
        print('Time data:\n%s' % (array(
            map(datetime.isoformat, self.timedata))))
        # original flags
        for col, value in self.origuniqueflags.items():
            print('Original flags in column %i:\n%s' % (col, ', '.join(value)))
        # resorting option
        for col, value in self.sortdict.items():
            print('Sort options in column %i:\n%s' % (
                col, ', '.join(
                    '%s --> %s' % item for item in value.items())))
        # redistribution option
        for rd in self.redistdata:
            rd.info()
        # final names
        print('---------------------------------------------')
        print('Final names in NetCDF file:\n' +
              ', '.join(self.finalnames))
        # restore numpy print options
        set_printoptions(**printops)

    def _set_cols(self, gridcols, time, defaultflags, cat, valcol):
        """function to set up column arrays.

        This function is called at initialization
        """
        self.defaultflagcols = defaultflags
        self.gridcols = gridcols
        self.valcol = valcol
        # set up columns which shall be concatenated
        self.catcolpairs = list(map(int, catcol.split(',')) for catcol in cat)
        self.catcols = list(chain(*self.catcolpairs))
        # columns which contain the flags
        self.flagcols = defaultflags+self.catcols
        # handling time
        if time is not None:
            # convert to dictionary
            self.time = {int(t): fmt for t, fmt in imap(
                lambda x: x.split(','), time)}
            self.timecols = sorted(self.time.keys())
            self.timefunc = self.gettimeindex
        else:
            self.time = {}
            self.timecols = []
            self.timefunc = self.dummytimeindex
        # all columns which shall be read from ifile
        self.usecols = sorted(gridcols + self.timecols + self.flagcols +
                              [valcol])

    def _read_data(self, ifile, **kwargs):
        """function to read data from text input file during initialization"""
        self.ifile = ifile
        self.data.ifiledata = loadtxt(ifile, dtype=str, delimiter='\t',
                                      usecols=self.usecols, unpack=True,
                                      **kwargs)

    def _read_grid(self, mask):
        """function to read in grid data from netCDF files during
        initialization"""
        # read mask data from mask file
        self.maskfile = mask.split(',')[0]
        with Dataset(mask.split(',')[0]) as nco:
            data = [
                nco.variables[varname][0, :] for varname in
                mask.split(',')[1:]]
            # convert masked arrays to normal arrays (much faster in loop)
            for idata in xrange(len(data)):
                if hasattr(data[idata], 'mask'):
                    data[idata] = data[idata].filled(missval)
            self.data.maskdata = data
            self.data.lon = nco.variables['lon'][:]
            self.data.lat = nco.variables['lat'][:]

    def read_alias(self, alias):
        # read alias grid data file. aliasdata is a list of numpy.ndarrays with
        # shape (2,n) where n is the number of the flags in the aliasfile.
        # Aliasdict is a dictionary with colums as keys and the converted
        # aliasdata
        self.aliasdict = {
            int(aliasitem.split(',')[0]): {
                key: val.astype(self.data.maskdata[self.gridcols.index(int(
                    aliasitem.split(',')[0]))].dtype)
                for key, val in roll(
                    loadtxt(aliasitem.split(',')[1], dtype=str, delimiter='\t',
                            usecols=[0, 1]), 1, axis=1)
                }
            for aliasitem in alias}

    def _set_up_timedata(self):
        # handling time
        if self.time != {}:
            self.timedata = unique(map(
                self.converttime,
                izip(*(self.data.ifiledata[self.usecols.index(tcol)]
                       for tcol in sorted(self.time.keys())))
                ))
            self.ntimes = len(self.timedata)
        else:
            self.timedata = array([datetime.now()])
            self.ntimes = 1

    def _read_sorting(self, sort, **kwargs):
        """function called at initialization to read sorting data from txt
        file"""
        # read 1d sorting data. sortdict is a dictionary with the column as key
        # for dictionaries with, again the flag as key and the alias as value
        self.sortdict = {
            int(sortitem.split(',')[0]): {
                flag: aliasflag for flag, aliasflag in
                loadtxt(sortitem.split(',')[1], dtype=str, delimiter='\t',
                        **kwargs)
                }
            for sortitem in sort}

    def _read_redist(self, redistribute):
        # read redistribution data. redistict is a dictionary which contains
        # the column as keys for dictionaries with, againg the keys 'flag'
        # for the name of the flag and a key 'data' for a dictionary which
        # contains the aliasflag as key and the 2-dimensional fraction data as
        # value
        self.redistdata = [Redistributer(sortitem) for sortitem in
                           redistribute]
        self.redistcols = unique(
            [sortitem.col for sortitem in self.redistdata]).tolist()
        self.redistflags = [sortitem.flag for sortitem in self.redistdata]

    def _set_names(self):
        """Method to set up final names for the variables and initializes
        final data arrays"""
        # get original flags
        self.origuniqueflags = {
            col: [flag for flag in
                  unique(self.data.ifiledata[self.usecols.index(col)])
                  if flag not in self.redistflags]
            for col in self.flagcols}
        # set up unique flags including sorted flags
        self.uniqueflags = {
            col: [flag for flag in
                  unique(self.data.ifiledata[self.usecols.index(col)])
                  if flag not in self.redistflags]
            for col in self.flagcols}
        self.uniqueflags.update({
            col: unique([self.sortdict[col][flag]
                         for flag in self.sortdict[col]]).tolist()
            for col in self.sortdict})
        # set up final names
        namesfromdefault = list(chain(*(chain(
            data for col, data in self.uniqueflags.items() if col not in
            self.catcols))))
        namesfromcatcols = list(chain(*(chain(
            flag for flag in map(
                lambda x: '_'.join(k for k in x),
                product(self.uniqueflags[col1], self.uniqueflags[col2])))
            for col1, col2 in self.catcolpairs)))
        self.finalnames = namesfromdefault + namesfromcatcols
        self.data.finaldata = {var: zeros([self.ntimes] +
                                          list(shape(self.data.maskdata[0])))
                               for var in self.finalnames}
        # we don't use masked arrays because much slower in the loop
        mask = self.data.maskdata[0] == missval
        for value in self.data.finaldata.values():
            for i in xrange(self.ntimes):
                value[i, :][mask] = missval

    def _set_add(self):
        """Method called during initialization to set up how data shall be
        added"""
        # dictionary containing the Adder instances for defaultcols which
        # determine where to add the value in finaldata for the given flag
        self.defaultadddict = {
            col: {flag: [
                Adder(
                    target=self.data.finaldata[self.sortdict.get(
                        col, {flag: flag}).get(flag, flag)])
                ] for flag in self.sortdict.get(col, self.uniqueflags[col])}
            for col in self.defaultflagcols}
        # dictionary containing the Adder instances for catcols which
        # determine where to add the value in finaldata for the given flag
        catadddict = {
            (col1, col2): {
                flagpair: [
                    Adder(self.data.finaldata['_'.join(self.sortdict.get(
                        [col1, col2][flagpair.index(flag)], {flag: flag})[flag]
                        for flag in flagpair)])]
                for flagpair in product(self.origuniqueflags[col1],
                                        self.origuniqueflags[col2])}
            for col1, col2 in self.catcolpairs}
        # now handle redistributed data
        for sortitem in self.redistdata:
            if sortitem.col in self.defaultflagcols:
                self.defaultadddict[sortitem.col][sortitem.flag] = [
                    Adder(target=self.data.finaldata[aliasflag],
                          mulc=sortitem.data[aliasflag])
                    for aliasflag in sortitem.data]
            elif sortitem.col in self.catcols:
                catcol = tuple(self.catcolpairs[sortitem.col in
                                                self.catcolpairs])
                sortitems = [
                    sortitem2 for sortitem2 in self.redistdata if sortitem2.col
                    in catcol and sortitem2 != sortitem]
                flags = []
                flags.insert(catcol.index(sortitem.col), [sortitem.flag])
                for col in catcol:
                    if col != sortitem.col and col not in self.sortdict.keys():
                        flags.insert(catcol.index(col), self.uniqueflags[col])
                    if col != sortitem.col and col in self.sortdict:
                        flags.insert(
                            catcol.index(col), unique(self.data.ifiledata[
                                self.usecols.index(col)]).tolist())
                try:
                    flags.remove([])
                except ValueError:
                    pass
                for flagpair in product(*flags):
                    catadddict[catcol][flagpair] = [0]*len(
                        sortitem.data.keys())
                    for i, replaceflag in izip(
                            xrange(len(sortitem.data.keys())),
                            sorted(sortitem.data.keys())):
                        newflagpair = list(flagpair)
                        for flag in newflagpair:
                            if flag == sortitem.flag:
                                newflagpair[newflagpair.index(flag)] = \
                                    replaceflag
                            elif catcol[flagpair.index(flag)] in self.sortdict:
                                newflagpair[newflagpair.index(flag)] = \
                                    self.sortdict[
                                        catcol[flagpair.index(flag)]][flag]
                        catadddict[catcol][flagpair][i] = Adder(
                            target=self.data.finaldata['_'.join(
                                flag for flag in newflagpair)],
                            mulc=sortitem.data[replaceflag])
        self.catadddict = catadddict

    def convert(self, verbose=False, maxrows=None):
        """Method to loop through the data and convert it"""
        # docstring is extended below
        dataslice = slice(0, maxrows)
        info = LoopInfo(total=len(self.data.ifiledata[0, dataslice]),
                        t0=datetime.now(), verbose=verbose)
        wrongvalues = 0
        wrongarea = 0

        for datatuple in izip(*self.data.ifiledata[:, dataslice]):
            info.info()
            fields = npall(
                # normal gridcols
                [self.data.maskdata[self.gridcols.index(col)] == datatuple[
                    self.usecols.index(col)].astype(
                        self.data.maskdata[self.gridcols.index(col)].dtype)
                    for col in self.gridcols if col not in self.aliasdict] +
                # alias cols
                [self.data.maskdata[self.gridcols.index(col)] ==
                    self.aliasdict[col][datatuple[
                        self.usecols.index(col)]] for col in self.aliasdict],
                axis=0)
            itime = self.timefunc([datatuple[self.usecols.index(col)]
                                   for col in sorted(self.time.keys())])
            for catcol in self.catadddict:
                for adderinstance in self.catadddict[catcol][tuple(datatuple[
                        self.usecols.index(col)] for col in catcol)]:
                    adderinstance.addfunc(
                        itime, fields, datatuple[
                            self.usecols.index(self.valcol)].astype(float))
            for col in self.defaultadddict:
                for adderinstance in self.defaultadddict[col][datatuple[
                        self.usecols.index(col)]]:
                    adderinstance.addfunc(
                        itime, fields,
                        datatuple[
                            self.usecols.index(self.valcol)].astype(float))
            if not npany(fields):
                wrongvalues += 1
                wrongarea += float(datatuple[self.usecols.index(self.valcol)])
        if verbose:
            print('\nNumber of wrong values: %i' % wrongvalues)
            print('Missed Area [ha]: %6.4f' % wrongarea)
            print('Missed Area: %1.3e %%' % (
                wrongarea/sum(self.data.ifiledata[
                    self.usecols.index(self.valcol)].astype(float))*100.))

    def output_nc(self, output, clobber=False, header=None, metadata={},
                  compression={}):
        """Method to create netCDF file out of final data
        """
        # docstring is extended below
        # set chunking parameter
        if os.path.exists(output):
            os.remove(output)
        if clobber is not False:
            if clobber == 'auto':
                clobber = chunk_shape_3D(
                    [self.ntimes] + list(self.data.maskdata[0].shape))
            nco = Dataset(output, 'w', format='NETCDF4_CLASSIC',
                          clobber=True)
        else:
            nco = Dataset(output, 'w', format='NETCDF4_CLASSIC')
        if not isinstance(compression, dict):
            compression = {'zlib': compression}
        if header is not None:
            nco.setncatts(header)
        nco.createDimension('time', None)
        nco.createDimension('lon', len(self.data.lon))
        nco.createDimension('lat', len(self.data.lat))

        timeo = nco.createVariable('time', 'f8', ('time'))
        timeo.standard_name = 'time'
        secondsperday = float(60*60*24)
        mystrftime = lambda x: (
            float(x.strftime('%Y%m%d')) +
            timedelta(hours=x.hour, minutes=x.minute, seconds=x.second,
                      microseconds=x.microsecond).seconds/secondsperday)
        timeo[:] = map(mystrftime, self.timedata)
        timeo.units = 'day as %Y%m%d.%f'

        lono = nco.createVariable("lon", "f4", ("lon"))
        lono.units = "degrees_east"
        lono.standard_name = "longitude"
        lono[:] = self.data.lon

        lato = nco.createVariable("lat", "f4", ("lat"))
        lato.units = "degrees_north"
        lato.standard_name = "latitude"
        lato[:] = self.data.lat
        for var, value in self.data.finaldata.items():
            if clobber is not False:
                varno = nco.createVariable(
                    var, "f4", ("time", "lat", "lon"),
                    chunksizes=clobber, fill_value=missval, **compression
                    )
            else:
                varno = nco.createVariable(
                    var, "f4", ("time", "lat", "lon"),
                    fill_value=missval, **compression
                    )
            for attr, val in metadata.get(var, {}).items():
                setattr(varno, attr, val)
            varno.standard_name = var
            varno[:] = value
        nco.close()

    def gettimeindex(self, data):
        return self.timedata.searchsorted(self.converttime(data))

    def dummytimeindex(self, data):
        # function which just returns the index 0
        return 0

    def converttime(self, data):
        # function to convert a data tuple into datetime instance
        return datetime.strptime(
            ','.join(data),
            ','.join([val for key, val in sorted(self.time.items())]))

    # ---- modify docstrings here ----
    __init__.__doc__ += '\nKeyword Arguments:\n  - ' + '\n  - '.join(
        ['%s: %s' % (key, val) for key, val in kwargdocs['init'].items()])
    convert.__doc__ += '\nKeyword Arguments:\n  - ' + '\n  - '.join(
        ['%s: %s' % (key, val) for key, val in kwargdocs['convert'].items()])
    output_nc.__doc__ += '\nKeyword Arguments:\n  - ' + '\n  - '.join(
        ['%s: %s' % (key, val) for key, val in kwargdocs['output'].items()])


# ---- automatic determination of chunking parameters -----
# functions taken from
# http://www.unidata.ucar.edu/staff/russ/public/chunk_shape_3D.py
# see also
"""
http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes
"""

import math
import operator


def binlist(n, width=0):
    """Return list of bits that represent a non-negative integer.

    n      -- non-negative integer
    width  -- number of bits in returned zero-filled list (default 0)
    """
    return map(int, list(bin(n)[2:].zfill(width)))


def numVals(shape):
    """Return number of values in chunk of specified shape, given by a list of
    dimension lengths.

    shape -- list of variable dimension sizes"""
    if(len(shape) == 0):
        return 1
    return reduce(operator.mul, shape)


def perturbShape(shape, onbits):
    """Return shape perturbed by adding 1 to elements corresponding to 1 bits
    in onbits

    shape  -- list of variable dimension sizes
    onbits -- non-negative integer less than 2**len(shape)
    """
    return map(sum, zip(shape, binlist(onbits, len(shape))))


def chunk_shape_3D(varShape, valSize=4, chunkSize=4096):
    """
    Return a 'good shape' for a 3D variable, assuming balanced 1D, 2D access

    varShape  -- length 3 list of variable dimension sizes
    chunkSize -- maximum chunksize desired, in bytes (default 4096)
    valSize   -- size of each data value, in bytes (default 4)

    Returns integer chunk lengths of a chunk shape that provides
    balanced access of 1D subsets and 2D subsets of a netCDF or HDF5
    variable var with shape (T, X, Y), where the 1D subsets are of the
    form var[:,x,y] and the 2D slices are of the form var[t,:,:],
    typically 1D time series and 2D spatial slices.  'Good shape' for
    chunks means that the number of chunks accessed to read either
    kind of 1D or 2D subset is approximately equal, and the size of
    each chunk (uncompressed) is no more than chunkSize, which is
    often a disk block size.
    """

    rank = 3  # this is a special case of n-dimensional function chunk_shape
    # ideal number of values in a chunk
    chunkVals = chunkSize / float(valSize)
    # ideal number of chunks
    numChunks = varShape[0]*varShape[1]*varShape[2] / chunkVals
    axisChunks = numChunks ** 0.25  # ideal number of chunks along each 2D axis
    cFloor = []  # will be first estimate of good chunk shape
    # cFloor  = [varShape[0] // axisChunks**2, varShape[1] // axisChunks,
    #     varShape[2] // axisChunks]
    # except that each chunk shape dimension must be at least 1
    # chunkDim = max(1.0, varShape[0] // axisChunks**2)
    if varShape[0] / axisChunks**2 < 1.0:
        chunkDim = 1.0
        axisChunks = axisChunks / math.sqrt(varShape[0]/axisChunks**2)
    else:
        chunkDim = varShape[0] // axisChunks**2
    cFloor.append(chunkDim)
    # factor to increase other dims if some must be increased to 1.0
    prod = 1.0
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            prod *= axisChunks / varShape[i]
    for i in range(1, rank):
        if varShape[i] / axisChunks < 1.0:
            chunkDim = 1.0
        else:
            chunkDim = (prod*varShape[i]) // axisChunks
        cFloor.append(chunkDim)

    # cFloor is typically too small, (numVals(cFloor) < chunkSize)
    # Adding 1 to each shape dim results in chunks that are too large,
    # (numVals(cCeil) > chunkSize).  Want to just add 1 to some of the
    # axes to get as close as possible to chunkSize without exceeding
    # it.  Here we use brute force, compute numVals(cCand) for all
    # 2**rank candidates and return the one closest to chunkSize
    # without exceeding it.
    bestChunkSize = 0
    cBest = cFloor
    for i in range(8):
        # cCand = map(sum,zip(cFloor, binlist(i, rank)))
        cCand = perturbShape(cFloor, i)
        thisChunkSize = valSize * numVals(cCand)
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list(cCand)  # make a copy of best candidate so far
    return map(int, cBest)


if __name__ == '__main__':
    main()
