import sys

def find_paths(paths,target,search_direction='backwards'):
    len_target = len(target)
    for i in range(0,len(paths)):
        if search_direction == 'backwards':
            if paths[i][len(paths[i])-len_target:len(paths[i])] == target:
                return i
        else:
            if paths[i][len(paths[i]):len(paths[i])+len_target] == target:
                return i
    sys.exit(target+' not found.')
