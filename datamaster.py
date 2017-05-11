#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-

#  _____        _        __  __           _
# |  __ \      | |      |  \/  |         | |
# | |  | | __ _| |_ __ _| \  / | __ _ ___| |_ ___ _ __
# | |  | |/ _` | __/ _` | |\/| |/ _` / __| __/ _ \ '__|
# | |__| | (_| | || (_| | |  | | (_| \__ \ ||  __/ |
# |_____/ \__,_|\__\__,_|_|  |_|\__,_|___/\__\___|_|

# A lab module loader for unified data analysis
#
# by Patrick Anker & Kaitlyn Morrell

import signal
import sys
import gc
import getopt
import os

# importing matplotlib so datamaster prints all requested plots at once
import matplotlib.pyplot as plt

cli_thread   = True # If false, the CLI terminates; goes false upon SIGKILL

LABS         = []   # List of all known labs given by subdirectories
current_labs = {}   # Key-value memory of active labs
selected_lab = None # Name of current selected lab; used for reloads

VERSION      = '1.1.0' # Current version of DataMaster

def fetch_lab(name, load):
    '''
    * Internal *

    Attempts to load a lab given an input name
    Returns the initialized lab module or None if an error is encountered
    '''
    obj = None

    if name not in LABS:
        # Directly return None instead of obj
        print('Invalid lab name')
        return None

    if name in current_labs and not load:
        if current_labs[name] is not None:
            print('%s loaded from memory' % (name))
            obj = current_labs[name]

    elif name in current_labs and load:
        try:
            unload_lab(name)

            obj = load_lab(name)

            if obj is None:
                print('Reload of %s failed; Try loading directly from file?' % (name))

                unload_lab(name)
                return None
            else:
                current_labs[name] = obj

        except Exception as err:
            print('Reload of lab failed')
            print(str(err))
    else:
        obj = load_lab(name)

        if obj is not None:
            current_labs[name] = obj
        else:
            print('Could not load \'%s\'' % (name))

    return obj

def load_lab(name):
    obj = None

    try:
        obj = __import__(name)

        if not hasattr(obj, 'lab'):
            print('__init__ file not properly configured.')
            obj = None
        else:
            current_labs[name] = obj
    except Exception as err:
        print('Lab analysis could not be loaded')
        print(str(err))

        # Resets the object reference in case something weird happened
        obj = None

    return obj

def unload_lab(name):
    '''
    * Internal *

    Removes all references of a lab from memory
    '''

    # Firstly, if the object has terminate(), call it; useful for freeing memory
    obj = current_labs[name]

    if hasattr(obj.lab, 'terminate') and callable(getattr(obj.lab, 'terminate')):
        getattr(obj.lab, 'terminate')()

    # Now, purge references
    rm = []

    global selected_lab

    if str(selected_lab) == name:
        selected_lab = None

    # Filter submodules that belong to target lab, if it exists
    for mod in sys.modules.keys():
        if mod.startswith('%s.' % (name)):
            rm.append(mod)

    for i in rm:
        sys.modules[i] = None # Maybe helps mark the VM to free memory?
        del sys.modules[i]

    # Redundant checks in case reference is broken
    if name in sys.modules:
        sys.modules[name] = None
        del sys.modules[name]

    if name in current_labs:
        del current_labs[name]

    gc.collect()

def select_lab(name, load=False):
    lab = fetch_lab(name, load)

    if lab is not None:
        global selected_lab
        selected_lab = name
        print('Selected Lab: %s\n' % (selected_lab))

def plot_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        if str(var) == 'all':

            for e in dir(obj.lab):
                if e.startswith('plot_') and callable(getattr(obj.lab, str(e))):
                    getattr(obj.lab, str(e))()

            return True

        try:
            getattr(obj.lab, ('plot_%s' % (var)))()
            return True
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()
        return False

def get_var(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        if str(var) == 'all':

            for e in dir(obj.lab):
                if e.startswith('get_') and callable(getattr(obj.lab, str(e))):
                    print(getattr(obj.lab, str(e))())

            return

        try:
            print(getattr(obj.lab, ('get_%s' % (var)))())
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()

def run_func(var):
    if selected_lab is not None:
        obj = current_labs[selected_lab]

        if str(var) == 'all': # Are you sure about that?

            for e in dir(obj.lab):
                if e.startswith('run_') and callable(getattr(obj.lab, str(e))):
                    getattr(obj.lab, str(e))()
            return

        try:
            getattr(obj.lab, ('run_%s' % (var)))()
        except Exception as err:
            print(str(err))
    else:
        print('No selected lab')
        usage()

def list():
    print('Available labs:')

    for s in LABS:
        if selected_lab is not None and s == selected_lab:
            print('> * %s' % (s))
        else:
            print('  * %s' % (s))

    if selected_lab is not None:
        obj   = current_labs[selected_lab]

        gets  = []
        plots = []
        runs  = []

        for e in dir(obj.lab):
            if e.startswith('get_') and callable(getattr(obj.lab, str(e))):
                gets.append(str(e).replace('get_', ''))
            elif e.startswith('plot_') and callable(getattr(obj.lab, str(e))):
                plots.append(str(e).replace('plot_', ''))
            elif e.startswith('run_') and callable(getattr(obj.lab, str(e))):
                runs.append(str(e).replace('run_', ''))

        if len(gets) != 0 or len(plots) != 0 or len(runs) != 0:
            print('----------------------------------')
            print('Functions for \'%s\'' % (selected_lab))

            if len(gets) != 0:
                print('Gets:')

                for g in gets:
                    print ('  * %s' % (g))

            if len(plots) != 0:
                print('Plots:')

                for p in plots:
                    print ('  * %s' % (p))

            if len(runs) != 0:
                print('Runnables:')

                for r in runs:
                    print ('  * %s' % (r))

def current_version():
    print('DataMaster version %s' % VERSION)

def usage():
    current_version()
    print('Usage: datamaster.py -s <name> [-g, -p] <data name>')
    print('\nCommands:\n  -h, --help: Prints out this help section')
    print('  -l, --list: Lists all the available labs and, if a lab is selected, all available gets and plots')
    print('  -s, --select <name>: Selects lab to compute data from')
    print('  -r, --reload: Reloads the selected lab from file')
    print('  -p, --plot <variable>: Calls a plotting function of the form \"plot_<variable>\"')
    print('  -g, --get <variable>: Prints out a value from function of the form \"get_<variable>\"')
    print('  -x, --run <variable>: Runs a custom function of the form \"run_<variable>\"')
    print('  -e, --exit: Explicit command to exit from DataMaster CLI')

def cli():
    legacy = False

    if sys.version_info < (3, 0):
        legacy = True

    while cli_thread:
        if legacy:
            # Py ≤ 2.7
            args = str(raw_input('> ')).split(' ')
            handle_args(args)
        else:
            # Py 3
            args = str(input('> ')).split(' ')
            handle_args(args)

def exit_handle(sig, frame):
    global cli_thread
    cli_thread = False # Safely halts while loop in thread

    print('\nExiting...')
    sys.exit(0)

def handle_args(args):
    try:
        opts, args = getopt.getopt(args, 'hlvs:rp:g:x:e', ['help', 'list', 'version', 'reload', 'select=', 'plot=', 'get=', 'run=', 'exit'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        return

    plotting = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            return

        elif opt in ('-l', '--list'):
            list()
            return

        elif opt in ('-v', '--version'):
            current_version()
            return

        elif opt in ('-s', '--select'):
            select_lab(arg)

        elif opt in ('-r', '--reload'):
            if selected_lab is not None:
                select_lab(selected_lab, True)
            else:
                print('No selected lab to reload')
                return

        elif opt in ('-p', '--plot'):
            if plot_var(arg) and plotting is False:
                plotting = True

        elif opt in ('-g', '--get'):
            get_var(arg)

        elif opt in ('-x', '--run'):
            run_func(arg)

        elif opt in ('-e', '--exit'):
            print('\nExiting...')
            sys.exit(0)

        else:
            usage()

    if plotting:
        plt.show()
        plt.close('all')

def main(argv):
    # Load in lab names
    global LABS
    LABS = [name for name in os.listdir('.') if os.path.isdir(name) and os.path.exists(os.path.join(name, 'lab.py'))]

    if len(argv) == 0:
        # register sigkill event and start looping CLI
        signal.signal(signal.SIGINT, exit_handle)
        cli()
    else:
        handle_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
