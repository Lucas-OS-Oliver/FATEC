#INTEGRANTES:#
#JONATA BERTOLONI DE VASCONCELOS  R.A.: 0791711024#
#LUCAS OLIVEIRA DE SOUZA  R.A.: 0791721002#
#IMPORTANDO IMAGEM(ENS) DO COMPUTADOR#
from google.colab import files
uploaded = files.upload()
#IMPORTANDO BIBLIOTECAS#
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
#BANCO COM 30 IMAGENS DISPONIBILIZADAS PELO PROFESSOR MURILO#
imagem = []
imagem.append("austin-distel-744oGeqpxPQ-unsplash.jpg")
imagem.append("anastasia-zhenina-fPX0XHxzCxI-unsplash.jpg")
imagem.append("austin-distel-EMPZ7yRZoGw-unsplash.jpg")
imagem.append("austin-distel-Jn1csk3lWDA-unsplash.jpg")
imagem.append("austin-distel-jpHw8ndwJ_Q-unsplash.jpg")
imagem.append("bruno-cervera-4i_9a-d_Q3E-unsplash.jpg")
imagem.append("chelsea-gates-faPPevPD0MQ-unsplash.jpg")
imagem.append("eggbank-ueAYDXll_N8-unsplash.jpg")
imagem.append("elle-hughes-ajTlDUwLuJk-unsplash.jpg")
imagem.append("erik-mclean-g6YNrpOq1Gk-unsplash.jpg")
imagem.append("eugene-zhyvchik-KUqJuDtrVkw-unsplash.jpg")
imagem.append("eugene-zhyvchik-u1gE6lvbloc-unsplash.jpg")
imagem.append("hiep-duong-cqspyPWxW_M-unsplash.jpg")
imagem.append("ilyuza-mingazova-2qMldY33AMo-unsplash.jpg")
imagem.append("isaac-quesada-1mvrY8osYkM-unsplash.jpg")
imagem.append("joshua-koblin-4hUxLunmxPM-unsplash.jpg")
imagem.append("lance-anderson-ixBBY-WuFRU-unsplash.jpg")
imagem.append("li-lin-0-MHQAG9XoA-unsplash.jpg")
imagem.append("marianna-berno-QK_ufggzGCk-unsplash.jpg")
imagem.append("nathan-anderson-0wKpv3o8D1I-unsplash.jpg")
imagem.append("neonbrand-AOJGuIJkoBc-unsplash.jpg")
imagem.append("neonbrand-JW6r_0CPYec-unsplash.jpg")
imagem.append("romain-b-Y0Mxn4xG4hA-unsplash.jpg")
imagem.append("surface-C389V--ZZrQ-unsplash.jpg")
imagem.append("surface-DQfIfnTuT3w-unsplash.jpg")
imagem.append("surface-O9m5k3_-iAs-unsplash.jpg")
imagem.append("thomas-de-luze-7xDfU-htISs-unsplash.jpg")
imagem.append("tim-motivv-AJCil71FOrc-unsplash.jpg")
imagem.append("tyler-nix-KFVsFjTTkBo-unsplash.jpg")
imagem.append("waldemar-brandt-FjAwyE8DLPw-unsplash.jpg")
escolha = int(input("Qual imagem deseja trabalhar? Escolha entre 0 e 29? "))
foto = cv2.imread(imagem[escolha])
cv2_imshow(foto)
largura = foto.shape[1]
altura = foto.shape[0]
escala = 0.00392
#LER OS NOMES DE CLASSE DO ARQUIVO DO TEXTO#
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
#GERAR DIFERENTES CORES PARA DIFERENTES CLASSES#
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
#LER O MODELO PRÉ-TREINADO E OS ARQUIVOS DE CONFIGURAÇÃO (WEIGHTS E CFG)#
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
#CRIAR BLOB DE ENTRADA#
blob = cv2.dnn.blobFromImage(foto, escala, (416,416), (0,0,0), True, crop=False)
#SETAR BLOB DE ENTRADA PARA A REDE#
net.setInput(blob)
#OBTER NOMES DAS CAMADAS DE SAÍDA#
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
#DESENHAR CAIXAS EM VOLTA DO OBJETO/PESSOAS DETECTADAS#
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 20)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 20)
#EXECUTAR ATRAVÉS DA REDE#
outs = net.forward(get_output_layers(net))
#RODANDO
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
#PARA CADA OBJETO/PESSOAS DETECTADAS DEVE CLASSIFICAR, CASO RECONHEÇA ACIMA DE 50%#
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * largura)
            center_y = int(detection[1] * altura)
            w = int(detection[2] * largura)
            h = int(detection[3] * altura)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#PASSAR POR TODAS AS OUTRAS DETECÇÕES E DESENHAR A CAIXA EM TORNO DO ALVO#
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(foto, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
#MOSTRAR IMAGEM COM DETECÇÃO#  
cv2_imshow(foto)
#LIBERA OS RECURSOS#
cv2.destroyAllWindows()
