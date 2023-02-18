import matplotlib.pyplot as plt
import numpy as np
train_data = [2.6100616455078125,2.5893056392669678,2.607635736465454]

test_data = [2.6182122230529785, 2.59806752204895, 2.616995334625244]

epochs = [500, 1000, 2000]

plt.figure(figsize=(12,6))
plt.title('Variation of NLL Vs # Epochs (3d_sin_5_5 dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(epochs, train_data, color='blue')#, linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(epochs, test_data, color='red')
plt.xticks([i for i in range(500,2001,500)],fontsize=20)
plt.yticks([i for i in np.arange(2.58,2.64,0.03)],fontsize=20)
plt.xlabel("# Epochs",fontsize=20)
plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)
lgd = plt.legend(['Train','Test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))
plt.savefig('NLL_q1_variation.eps',  bbox_inches='tight')
plt.savefig('NLL_q1_variation.png', bbox_inches='tight')
plt.show()
