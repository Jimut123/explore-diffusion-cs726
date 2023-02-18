import matplotlib.pyplot as plt
import numpy as np

train_data_S = [3.4212019443511963, 3.4120359420776367, 3.4158198833465576]
test_data_S = [3.429100751876831, 3.4030673503875732, 3.4058990478515625]

train_data_H = [3.4615492820739746, 3.444607973098755, 3.4666154384613037]
test_data_H = [3.4607110023498535, 3.4455697536468506, 3.4650495052337646]



model = ['A', 'B', 'C']

plt.figure(figsize=(12,6))
plt.title('Variation of NLL across different models', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(model, train_data_S, color='blue', linewidth=3.0)
plt.plot(model, test_data_S, color='red', linewidth=3.0)

plt.plot(model, train_data_H, color='blue', linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(model, test_data_H, color='red', linestyle='-.', dashes=(5, 1), linewidth=3.0)


# plt.xticks([i for i in range(500,2001,500)],fontsize=20)
plt.yticks([i for i in np.arange(3.39,3.53,0.03)],fontsize=20)
plt.xlabel("Models for comparison",fontsize=20)
plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
lgd = plt.legend(['3d_sin_5_5 train','3d_sin_5_5 test', 'helix_3D train', 'helix_3D test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))
plt.savefig('q2.eps',  bbox_inches='tight')
plt.savefig('q2.png', bbox_inches='tight')
plt.show()
