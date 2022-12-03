from . import models
from django.shortcuts import render

def uploadFile(request):
    if request.method == "POST":
        # Fetching the form data
        fileTitle = request.POST["fileTitle"]
        uploadedFile = request.FILES["uploadedFile"]

        # Saving the information in the database
        document = models.Document(
            title = fileTitle,
            uploadedFile = uploadedFile
        )
        document.save()
 #      업로드할때마다 images에 있는 파일들 모듈돌림 (220525)
 #      import ai_module

    documents = models.Document.objects.all()

    return render(request, "Core/upload-file.html", context = {
        "files": documents
    })
