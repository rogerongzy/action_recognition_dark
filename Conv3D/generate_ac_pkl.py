from fileinput import close
import pickle


def read_pkl(path):
    f = open(path,'rb')
    
    content = pickle.load(f) # list
    
    # print(content.index('CliffDiving'))
    print(content)


def write_pkl(path):
    action_list = ['Drink', 'Jump', 'Pick', 'Pour', 'Push', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
    f = open(path, 'wb')
    pickle.dump(action_list, f)
    f.close()
    return 0



if __name__ == "__main__":
    read_pkl('action.pkl') # 'UCF101actions.pkl' 'action.pkl'
    # write_pkl('action.pkl')