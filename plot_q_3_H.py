import matplotlib.pyplot as plt
import numpy as np

train_data = [2.3226993083953857, 2.258214235305786, 2.25455904006958, 2.2538709926605225, 2.2560789585113525]
test_data = [2.3248841762542725, 2.2616653442382812,  2.2563207149505615, 2.2564606952667236, 2.2589921951293945 ]



t_steps = [10, 50, 100, 150, 200]


plt.figure(figsize=(12,6))
plt.title('Variation of NLL Vs #Time Steps (helix_3D dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(t_steps, train_data, color='blue')#, linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(t_steps, test_data, color='red')
plt.xticks([i for i in range(0,201,20)],fontsize=20)
plt.yticks([i for i in np.arange(2.24,2.35,0.04)],fontsize=20)
plt.xlabel("Time Steps",fontsize=20)

plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Train','Test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))

plt.savefig('NLL_q3_var_t_step_H.eps',  bbox_inches='tight')
plt.savefig('NLL_q3_var_t_step_H.png', bbox_inches='tight')

plt.show()
