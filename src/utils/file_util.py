import os

# check whether the dir is exist
def Exists(dir):
    return os.path.exists(dir)


# delete the dir recursively
def DeleteRecursively(dir):
    dir = dir.replace('\\', '/')
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            DeleteRecursively(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)


# make a dir
def MakeDirs(dir):
    os.mkdir(dir)
