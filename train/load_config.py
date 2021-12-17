import pickle



file = open('/home/mkoch/projects/EfficientCN/experiments/effcn_smallnorb_2021_12_14_21_43_34/config.pkl', 'rb')
config = pickle.load(file)
file.close()


file = open('/home/mkoch/projects/EfficientCN/experiments/effcn_smallnorb_2021_12_14_21_43_34/stats.pkl', 'rb')
stats = pickle.load(file)
file.close()




print(config)
print(stats)