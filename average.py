import json

filename = '/home/bringascastle/Documentos/repos/SSD/results/results.json'

with open(filename, "r") as file:
    datos = json.load(file)


def parserFloat(array):
    return [float(v) for v in array] 

print("NMS: ", len(datos['NMS']))
print("El promedio NMS: {}".format(sum(parserFloat(datos['NMS'][100:]))/len(datos['NMS'][100:])))

print("DRAW: ", len(datos['DRAW']))
print("El promedio draw: {}".format(sum(parserFloat(datos['DRAW'][100:]))/len(datos['DRAW'][100:])))

print("GENERAL: ", len(datos['GENERAL']))
print("El promedio general: {}".format(sum(parserFloat(datos['GENERAL'][100:]))/len(datos['GENERAL'][100:])))

print("RED: ", len(datos['RED']))
print("El promedio red: {}".format(sum(parserFloat(datos['RED'][100:]))/len(datos['RED'][100:])))