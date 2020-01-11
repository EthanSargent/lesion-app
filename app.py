from flask import Flask, request, render_template, Response
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import os
import time
from flask import Response
from torchvision import models
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from textwrap import wrap

app = Flask(__name__)

classes = ['actinic keratosis',
 'basal cell carcinoma',
 'dermatofibroma',
 'melanoma',
 'nevus',
 'pigmented benign\nkeratosis',
 'seborrheic keratosis',
 'squamous cell carcinoma',
 'vascular lesion']

model = models.densenet161(pretrained=True)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, len(classes))
model.load_state_dict(torch.load('models/production_dict.pt'))

criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
model.to(device)

classes = [ '\n'.join(wrap(l, 10)) for l in classes ]

@app.route('/')  # a decorator wraps a function, modifying its behavior
def index():
	return render_template('index.html')

@app.route('/upload_img', methods=['GET','POST'])
def upload_img():
	img = Image.open(request.files['pic'].stream)
	x = TF.to_tensor(img)
	x.unsqueeze_(0)
	with torch.set_grad_enabled(False):
		ans = model(x.to(device))
	
	sm = nn.Softmax(dim=1)
	preds = sm(ans).cpu().numpy()[0]

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.barh(range(9), np.log(preds))

	ax.set_yticks(range(9))
	ax.set_yticklabels(['{}'.format(lc) for lc in classes])
	ax.set_title('Prediction class (log proba)')
	fig.tight_layout(pad=1)

	name = 'diagnosis' + str(time.time()).replace('.','') + '.png'
	fig.savefig('static/' + name)
	plt.clf()

	if np.argmax([preds])==4:
		diagnosis = 'benign'
	else:
		diagnosis = classes[np.argmax([preds])]
	proba = preds[np.argmax([preds])]

	return render_template('diag.html', newPlotName=name, diagnosis=diagnosis, proba=str(proba))
