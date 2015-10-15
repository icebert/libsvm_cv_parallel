#!/usr/bin/env python

import os, sys, traceback, getpass, time, re, shlex
from mpi4py import MPI

if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue


comm = MPI.COMM_WORLD
total_mpi = comm.Get_size()
rank_id   = comm.Get_rank()




class Worker:
    def __init__(self, options, cexp, gexp):
        self.options = options
        self.cexp = cexp
        self.gexp = gexp
    
    def get_cmd(self, c, g):
        cmdline = '"' + self.options["svmtrain_pathname"] + '"'
        if self.options["grid_with_c"]: 
            cmdline += ' -c {0} '.format(c)
        if self.options["grid_with_g"]: 
            cmdline += ' -g {0} '.format(g)
        cmdline += ' -v {0} {1} {2} '.format\
            (self.options["fold"], self.options["pass_through_string"], self.options["dataset_pathname"])
        return cmdline
    
    def run(self):
        c, g = None, None
        if self.cexp != None:
            c = 2.0**self.cexp
        if self.gexp != None:
            g = 2.0**self.gexp
        cmdline = self.get_cmd(c,g)
        cmd = shlex.split(cmdline)
        path = cmd[0]
        input = cmd[-1]
        param = ' '.join(cmd[1:-1])
        sys.path.append(path)
        from svmutil import svm_read_problem, svm_train
        y, x = svm_read_problem(input)
        cv_acc = svm_train(y, x, param)
        return (self.cexp, self.gexp, cv_acc)



if rank_id == 0:
    class GridOption((object)):
        def __init__(self, dataset_pathname, options):
            dirname = os.path.dirname(__file__)
            if sys.platform != 'win32':
                self.svmtrain_pathname = os.path.join(dirname, '../svm-train')
                self.gnuplot_pathname = '/usr/bin/gnuplot'
            else:
                # example for windows
                self.svmtrain_pathname = os.path.join(dirname, r'..\windows\svm-train.exe')
                # svmtrain_pathname = r'c:\Program Files\libsvm\windows\svm-train.exe'
                self.gnuplot_pathname = r'c:\tmp\gnuplot\binary\pgnuplot.exe'
            self.fold = 5
            self.c_begin, self.c_end, self.c_step = -5,  15,  2
            self.g_begin, self.g_end, self.g_step =  3, -15, -2
            self.grid_with_c, self.grid_with_g = True, True
            self.dataset_pathname = dataset_pathname
            self.dataset_title = os.path.split(dataset_pathname)[1]
            self.out_pathname = '{0}.out'.format(self.dataset_title)
            self.png_pathname = '{0}.png'.format(self.dataset_title)
            self.pass_through_string = ' '
            self.resume_pathname = None
            self.parse_options(options)

        def parse_options(self, options):
            if type(options) == str:
                options = options.split()
            i = 0
            pass_through_options = []
            
            while i < len(options):
                if options[i] == '-log2c':
                    i = i + 1
                    if options[i] == 'null':
                        self.grid_with_c = False
                    else:
                        self.c_begin, self.c_end, self.c_step = map(float,options[i].split(','))
                elif options[i] == '-log2g':
                    i = i + 1
                    if options[i] == 'null':
                        self.grid_with_g = False
                    else:
                        self.g_begin, self.g_end, self.g_step = map(float,options[i].split(','))
                elif options[i] == '-v':
                    i = i + 1
                    self.fold = options[i]
                elif options[i] in ('-c','-g'):
                    raise ValueError('Use -log2c and -log2g.')
                elif options[i] == '-svmtrain':
                    i = i + 1
                    self.svmtrain_pathname = options[i]
                elif options[i] == '-gnuplot':
                    i = i + 1
                    if options[i] == 'null':
                        self.gnuplot_pathname = None
                    else:   
                        self.gnuplot_pathname = options[i]
                elif options[i] == '-out':
                    i = i + 1
                    if options[i] == 'null':
                        self.out_pathname = None
                    else:
                        self.out_pathname = options[i]
                elif options[i] == '-png':
                    i = i + 1
                    self.png_pathname = options[i]
                elif options[i] == '-resume':
                    if i == (len(options)-1) or options[i+1].startswith('-'):
                        self.resume_pathname = self.dataset_title + '.out'
                    else:
                        i = i + 1
                        self.resume_pathname = options[i]
                else:
                    pass_through_options.append(options[i])
                i = i + 1

            self.pass_through_string = ' '.join(pass_through_options)
            if not os.path.exists(self.svmtrain_pathname):
                raise IOError('svm-train executable not found')
            if not os.path.exists(self.dataset_pathname):
                raise IOError('dataset not found')
            if self.resume_pathname and not os.path.exists(self.resume_pathname):
                raise IOError('file for resumption not found')
            if not self.grid_with_c and not self.grid_with_g:
                raise ValueError('-log2c and -log2g should not be null simultaneously')
            if self.gnuplot_pathname and not os.path.exists(self.gnuplot_pathname):
                sys.stderr.write('gnuplot executable not found\n')
                self.gnuplot_pathname = None
    
    
    
    def redraw(db,best_param,gnuplot,options,tofile=False):
        if len(db) == 0: return
        begin_level = round(max(x[2] for x in db)) - 3
        step_size = 0.5

        best_log2c,best_log2g,best_rate = best_param

        # if newly obtained c, g, or cv values are the same,
        # then stop redrawing the contour.
        if all(x[0] == db[0][0]  for x in db): return
        if all(x[1] == db[0][1]  for x in db): return
        if all(x[2] == db[0][2]  for x in db): return

        if tofile:
            gnuplot.write(b"set term png transparent small linewidth 2 medium enhanced\n")
            gnuplot.write("set output \"{0}\"\n".format(options.png_pathname.replace('\\','\\\\')).encode())
            #gnuplot.write(b"set term postscript color solid\n")
            #gnuplot.write("set output \"{0}.ps\"\n".format(options.dataset_title).encode().encode())
        elif sys.platform == 'win32':
            gnuplot.write(b"set term windows\n")
        else:
            gnuplot.write( b"set term x11\n")
        gnuplot.write(b"set xlabel \"log2(C)\"\n")
        gnuplot.write(b"set ylabel \"log2(gamma)\"\n")
        gnuplot.write("set xrange [{0}:{1}]\n".format(options.c_begin,options.c_end).encode())
        gnuplot.write("set yrange [{0}:{1}]\n".format(options.g_begin,options.g_end).encode())
        gnuplot.write(b"set contour\n")
        gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
        gnuplot.write(b"unset surface\n")
        gnuplot.write(b"unset ztics\n")
        gnuplot.write(b"set view 0,0\n")
        gnuplot.write("set title \"{0}\"\n".format(options.dataset_title).encode())
        gnuplot.write(b"unset label\n")
        gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  accuracy = {2}%\" \
                      at screen 0.5,0.85 center\n". \
                      format(best_log2c, best_log2g, best_rate).encode())
        gnuplot.write("set label \"C = {0}  gamma = {1}\""
                      " at screen 0.5,0.8 center\n".format(2**best_log2c, 2**best_log2g).encode())
        gnuplot.write(b"set key at screen 0.9,0.9\n")
        gnuplot.write(b"splot \"-\" with lines\n")
        
        db.sort(key = lambda x:(x[0], -x[1]))

        prevc = db[0][0]
        for line in db:
            if prevc != line[0]:
                gnuplot.write(b"\n")
                prevc = line[0]
            gnuplot.write("{0[0]} {0[1]} {0[2]}\n".format(line).encode())
        gnuplot.write(b"e\n")
        gnuplot.write(b"\n") # force gnuplot back to prompt when term set failure
        gnuplot.flush()


    def calculate_jobs(options):
        
        def range_f(begin,end,step):
            # like range, but works on non-integer too
            seq = []
            while True:
                if step > 0 and begin > end: break
                if step < 0 and begin < end: break
                seq.append(begin)
                begin = begin + step
            return seq
        
        def permute_sequence(seq):
            n = len(seq)
            if n <= 1: return seq
        
            mid = int(n/2)
            left = permute_sequence(seq[:mid])
            right = permute_sequence(seq[mid+1:])
        
            ret = [seq[mid]]
            while left or right:
                if left: ret.append(left.pop(0))
                if right: ret.append(right.pop(0))
                
            return ret  

        
        c_seq = permute_sequence(range_f(options.c_begin,options.c_end,options.c_step))
        g_seq = permute_sequence(range_f(options.g_begin,options.g_end,options.g_step))

        if not options.grid_with_c:
            c_seq = [None]
        if not options.grid_with_g:
            g_seq = [None] 
        
        nr_c = float(len(c_seq))
        nr_g = float(len(g_seq))
        i, j = 0, 0
        jobs = []

        while i < nr_c or j < nr_g:
            if i/nr_c < j/nr_g:
                # increase C resolution
                line = []
                for k in range(0,j):
                    line.append((c_seq[i],g_seq[k]))
                i = i + 1
                jobs.append(line)
            else:
                # increase g resolution
                line = []
                for k in range(0,i):
                    line.append((c_seq[k],g_seq[j]))
                j = j + 1
                jobs.append(line)

        resumed_jobs = {}
        
        if options.resume_pathname is None:
            return jobs, resumed_jobs

        for line in open(options.resume_pathname, 'r'):
            line = line.strip()
            rst = re.findall(r'rate=([0-9.]+)',line)
            if not rst: 
                continue
            rate = float(rst[0])

            c, g = None, None 
            rst = re.findall(r'log2c=([0-9.-]+)',line)
            if rst: 
                c = float(rst[0])
            rst = re.findall(r'log2g=([0-9.-]+)',line)
            if rst: 
                g = float(rst[0])

            resumed_jobs[(c,g)] = rate

        return jobs, resumed_jobs
    
    def update_param(options, c,g,rate,best_c,best_g,best_rate,worker,resumed):
        if (rate > best_rate) or (rate==best_rate and g==best_g and c<best_c):
            best_rate,best_c,best_g = rate,c,g
        stdout_str = '[{0}] {1} {2} (best '.format\
            (worker,' '.join(str(x) for x in [c,g] if x is not None),rate)
        output_str = ''
        if c != None:
            stdout_str += 'c={0}, '.format(2.0**best_c)
            output_str += 'log2c={0} '.format(c)
        if g != None:
            stdout_str += 'g={0}, '.format(2.0**best_g)
            output_str += 'log2g={0} '.format(g)
        stdout_str += 'rate={0})'.format(best_rate)
        print(stdout_str)
        if options.out_pathname and not resumed:
            output_str += 'rate={0}\n'.format(rate)
            result_file.write(output_str)
            result_file.flush()
        
        return best_c,best_g,best_rate
    
    
    def find_parameters(dataset_pathname, options=''):
        options = GridOption(dataset_pathname, options);

        if options.gnuplot_pathname:
            gnuplot = Popen(options.gnuplot_pathname,stdin = PIPE,stdout=PIPE,stderr=PIPE).stdin
        else:
            gnuplot = None
            
        # put jobs in queue

        jobs,resumed_jobs = calculate_jobs(options)
        job_queue = Queue(0)
        result_queue = Queue(0)

        for (c,g) in resumed_jobs:
            result_queue.put(('resumed',c,g,resumed_jobs[(c,g)]))

        for line in jobs:
            for (c,g) in line:
                if (c,g) not in resumed_jobs:
                    job_queue.put((c,g))

        # hack the queue to become a stack --
        # this is important when some thread
        # failed and re-put a job. It we still
        # use FIFO, the job will be put
        # into the end of the queue, and the graph
        # will only be updated in the end
     
        job_queue._put = job_queue.queue.appendleft
        
        
        # Broadcast the options
        for i in range(1, total_mpi):
            comm.send(options.__dict__, dest=i, tag=0)
        
        (cexp0, gexp0) = job_queue.get()
        
        # Dispatch the jobs
        id = 0
        rnd = [0] * total_mpi
        while not job_queue.empty():
            id = (id + 1) % total_mpi
            if id == 0:
                continue
            rnd[id] += 1
            (cexp, gexp) = job_queue.get()
            comm.isend((cexp, gexp), dest=id, tag=rnd[id])
        
        for i in range(1, total_mpi):
            comm.isend((None, None), dest=i, tag=(rnd[i]+1))
        
        
        worker = Worker(options.__dict__, cexp0, gexp0)
        (cexp, gexp, rate) = worker.run()
        result_queue.put(('0', cexp, gexp, rate))
        
        for id in range(1, total_mpi):
            for i in range(1, rnd[id]+1):
                (workerid, cexp, gexp, rate) = comm.recv(source=id, tag=i)
                result_queue.put((workerid, cexp, gexp, rate))
        
        best_rate = -1
        best_c,best_g = None, None
        while not result_queue.empty():
            (workerid, c, g, rate) = result_queue.get()
            best_c, best_g, best_rate = update_param(options, c, g, rate, best_c, best_g, best_rate, 'mpi'+workerid, False)
            if gnuplot and options.grid_with_c and options.grid_with_g:
                redraw(db,[best_c, best_g, best_rate],gnuplot,options)
                redraw(db,[best_c, best_g, best_rate],gnuplot,options,True)
        
        if options.out_pathname:
            result_file.close()
        best_param, best_cg  = {}, []
        if best_c != None:
            best_param['c'] = 2.0**best_c
            best_cg += [2.0**best_c]
        if best_g != None:
            best_param['g'] = 2.0**best_g
            best_cg += [2.0**best_g]
        print('{0} {1}'.format(' '.join(map(str,best_cg)), best_rate))
        
        return best_rate, best_param
    
    
    
    
#
# Main
#
    def exit_with_help():
        print("""\
Usage: grid.py [grid_options] [svm_options] dataset

grid_options :
-log2c {begin,end,step | "null"} : set the range of c (default -5,15,2)
    begin,end,step -- c_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with c
-log2g {begin,end,step | "null"} : set the range of g (default 3,-15,-2)
    begin,end,step -- g_range = 2^{begin,...,begin+k*step,...,end}
    "null"         -- do not grid with g
-v n : n-fold cross validation (default 5)
-svmtrain pathname : set svmutil.py executable path
-gnuplot {pathname | "null"} :
    pathname -- set gnuplot executable path and name
    "null"   -- do not plot 
-out {pathname | "null"} : (default dataset.out)
    pathname -- set output file path and name
    "null"   -- do not output file
-png pathname : set graphic output file path and name (default dataset.png)
-resume [pathname] : resume the grid task using an existing output file (default pathname is dataset.out)
    This is experimental. Try this option only if some parameters have been checked for the SAME data.

svm_options : additional options for svm-train""")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        exit_with_help()
    dataset_pathname = sys.argv[-1]
    options = sys.argv[1:-1]
    try:
        find_parameters(dataset_pathname, options)
    except (IOError,ValueError) as e:
        sys.stderr.write(str(e) + '\n')
        sys.stderr.write('Try "grid.py" for more information.\n')
        sys.exit(1)

#
# Worker Processes
#
else:
    options = comm.recv(source=0, tag=0)
    rnd = 0
    while True:
        rnd += 1
        (cexp, gexp) = comm.recv(source=0, tag=rnd)
        if cexp is None and gexp is None:
            break
        
        worker = Worker(options, cexp, gexp)
        (cexp, gexp, rate) = worker.run()
        
        comm.isend((str(rank_id), cexp, gexp, rate), dest=0, tag=rnd)



