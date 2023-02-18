import matplotlib.pyplot as plt
import numpy as np

train_data_sigmoid = [2.6103525161743164, 2.600022077560425, 2.70386004447937]
test_data_sigmoid = [2.6182682514190674, 2.610560894012451, 2.7158315181732178]

train_data_linear = [2.6100616455078125,2.612828254699707,2.700244426727295]
test_data_linear = [2.6182122230529785,2.622276782989502,2.709900140762329]

train_data_quad = [2.6206560134887695,2.5966806411743164,2.7205965518951416]
test_data_quad = [2.628101348876953,2.6076087951660156,2.7317330837249756]



schedule_beta = ["l_beta=1e-05,u_beta=1e-02", "l_beta=1e-07,u_beta=1e-01", "l_beta=1e-07,u_beta=1e-03"]





plt.figure(figsize=(12,6))
plt.title('Variation of schedulers (3d_sin_5_5 dataset)', fontsize=20, fontname = 'DejaVu Serif', fontweight = 500)

plt.plot(schedule_beta, train_data_sigmoid, color='blue')
plt.plot(schedule_beta, test_data_sigmoid, color='red')

plt.plot(schedule_beta, train_data_quad, color='green')
plt.plot(schedule_beta, test_data_quad, color='black',linewidth=3.0)

plt.plot(schedule_beta, train_data_linear, color='purple')
plt.plot(schedule_beta, test_data_quad, color='orange')


plt.xticks(fontsize=20,rotation = 45)
plt.yticks([i for i in np.arange(2.58,2.75,0.04)],fontsize=20)
plt.xlabel("Variation of Schedulers",fontsize=20)

plt.ylabel("Neg. Log Likelihood",fontsize=20)
plt.yticks(fontsize=20)

lgd = plt.legend(['Train Sigmoid','Test Sigmoid', 'Train Quad','Test Quad', 'Train Linear','Test Linear'],loc="upper left",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.53, 0.67))

plt.savefig('NLL_q4_sched_S.eps',  bbox_inches='tight')
plt.savefig('NLL_q4_sched_S.png', bbox_inches='tight')

plt.show()
