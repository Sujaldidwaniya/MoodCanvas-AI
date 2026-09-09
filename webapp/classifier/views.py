import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml_model import predict_image

def index(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        file_path = fs.path(filename)
        file_url = fs.url(filename)

        predicted_class, confidence, all_scores = predict_image(file_path)

        context = {
            'prediction': predicted_class,
            'confidence': confidence,
            'all_scores': all_scores,
            'image_url': file_url,
        }

    return render(request, 'classifier/index.html', context)