import matplotlib.pyplot as plt
import numpy as np

train_data = [2.7164673805236816,2.629031181335449,2.6100616455078125]
test_data = [2.7049787044525146,2.6372499465942383,2.6182122230529785]



t_steps = [10, 50, 100, 150, 200]


plt.figure(figsize=(12,6))
plt.title('Variation of NLL Vs #Time Steps (3d_sin_5_5 dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(t_steps, train_data, color='blue')#, linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(t_steps, test_data, color='red')
plt.xticks([i for i in range(500,2001,500)],fontsize=20)
plt.yticks([i for i in np.arange(3.44,3.48,0.02)],fontsize=20)
plt.xlabel("Time Steps",fontsize=20)

plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Train','Test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))

plt.savefig('NLL_q3_var_t_step_S.eps',  bbox_inches='tight')
plt.savefig('NLL_q3_var_t_step_S.png', bbox_inches='tight')

plt.show()
