import matplotlib.pyplot as plt
import numpy as np
train_data = [3.4714741706848145, 3.444607973098755, 3.468149185180664]

test_data = [3.4721148014068604, 3.4455697536468506, 3.4654059410095215]

epochs = [500, 1000, 2000]


plt.figure(figsize=(12,6))
plt.title('Variation of NLL Vs # Epochs (helix_3D dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(epochs, train_data, color='blue')#, linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(epochs, test_data, color='red')
plt.xticks([i for i in range(500,2001,500)],fontsize=20)
plt.yticks([i for i in np.arange(3.44,3.48,0.02)],fontsize=20)
plt.xlabel("# Epochs",fontsize=20)

plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Train','Test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))
plt.savefig('NLL_q1_variation_H.eps',  bbox_inches='tight')
plt.savefig('NLL_q1_variation_H.png', bbox_inches='tight')

plt.show()
