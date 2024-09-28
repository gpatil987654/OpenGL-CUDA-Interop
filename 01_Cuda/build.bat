cls

del *.obj
del *.exe
del *.txt

nvcc.exe -c -o SineWave.cu.obj SineWave.cu

cl.exe /c /EHsc /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" /I ..\Include\ *.cpp

rc.exe /I ..\Include\ ../resource/ogl.rc 

link *.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" /LIBPATH:..\lib ../resource/ogl.res user32.lib gdi32.lib kernel32.lib /OUT:..\bin\%1.exe

del *.obj

..\bin\%1.exe
