from calendar import EPOCH
import json
import matplotlib.pyplot as plt


epoch = 232

#filename2 = '/home/bringascastle/Escritorio/resultados_ssd_lite_transfer/losses.json'
#filename1 = '//home/bringascastle/Escritorio/resultados_ssd_lite_finetunning/losses.json'
filename3 = '/home/bringascastle/Documentos/repos/SSD/results/loss_19.json'

# with open(filename1, "r") as file:
#     finetunning = json.load(file)
#     # 2. Update json object
# with open(filename2, "r") as file:
#     transfer = json.load(file)
#     # 2. Update json object
# with open(filename3, "r") as file:
#     fromscratch = json.load(file)
#     # 2. Update json object

# def getArray(data):
#     a = []
#     i = 0
#     for i in range(0, len(data)):#, 4132):
#         a.append(float(data[i]))
#         if i == 231:
#             break
#         i += 1
#     return a

epoch = range(1, epoch , 1)

a= [2.0601, 1.4285, 1.2931, 1.2162, 1.1681, 1.1329, 1.1188, 1.0910, 1.0706, 1.0645, 1.0556, 1.0437, 1.0373, 1.0364, 1.0241, 1.0176, 1.0213, 1.0122, 1.0045, 1.0168, 1.0016, 0.9986, 1.0009, 0.9965, 0.9990, 0.9921, 0.9901, 0.9889, 0.9886, 0.9846]

epoch = range(1, len(a) + 1, 1)
plt.plot ( epoch, a, 'r', label='Loss' )
plt.title ('Perdida SSD artesanal')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()


