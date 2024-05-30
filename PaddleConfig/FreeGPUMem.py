import os

##################################################################
#
#   此文件用于清理显存和系统共享内存
#
##################################################################

result = os.popen("fuser -v /dev/nvidia*").read()
results = result.split()
for pid in results:
    os.system(f"kill -9 {int(pid)}")
os.system("rm -rf /dev/shm/*")
