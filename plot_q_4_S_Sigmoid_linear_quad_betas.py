import matplotlib.pyplot as plt
import numpy as np

train_data_sigmoid = [2.6103525161743164, 2.600022077560425, ]
test_data_sigmoid = [2.6182682514190674, 2.610560894012451, ]

train_data_linear = [,,]
test_data_linear = [,,]

train_data_quad = [,,]
test_data_quad = [,,]



schedule_beta = ["l_beta=1e-05,u_beta=1e-02", "l_beta=1e-07,u_beta=1e-01", "l_beta=1e-07,u_beta=1e-03"]





plt.figure(figsize=(12,6))
plt.title('Variation of schedulers (3d_sin_5_5 dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(t_steps, train_data, color='blue')#, linestyle='-.', dashes=(5, 1), linewidth=3.0)
plt.plot(t_steps, test_data, color='red')
plt.xticks([i for i in range(0,201,20)],fontsize=20)
plt.yticks([i for i in np.arange(2.6,2.75,0.04)],fontsize=20)
plt.xlabel("Time Steps",fontsize=20)

plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Train','Test'],loc="upper right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))

plt.savefig('NLL_q3_var_t_step_S.eps',  bbox_inches='tight')
plt.savefig('NLL_q3_var_t_step_S.png', bbox_inches='tight')

plt.show()
