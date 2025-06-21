#!/usr/bin/env python3

import os

Import('envcontrib')

envPyCuAmpcor = envcontrib.Clone()
package = envPyCuAmpcor['PACKAGE']
project = 'pycuampcor'
envPyCuAmpcor['PROJECT'] = project

Export('envPyCuAmpcor')

if envPyCuAmpcor['GPU_ACC_ENABLED']:
    envPyCuAmpcor.Append(CPPPATH=envPyCuAmpcor['CUDACPPPATH'])
    envPyCuAmpcor.Append(LIBPATH=envPyCuAmpcor['CUDALIBPATH'])
    envPyCuAmpcor.Append(LIBS=['cudart','cufft','cublas'])

    cudaScons = os.path.join('src', 'SConscript')
    SConscript(cudaScons, variant_dir=os.path.join(envPyCuAmpcor['PRJ_SCONS_BUILD'], package, project, 'src'))

    install = os.path.join(envPyCuAmpcor['PRJ_SCONS_INSTALL'],package,project)
    initFile = 'pycuampcor/__init__.py'

    listFiles = [initFile]
    envPyCuAmpcor.Install(install, listFiles)
    envPyCuAmpcor.Alias('install', install)
