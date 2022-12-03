from django.http import JsonResponse
from django.views import View
import ai_module
from importlib import reload

def api_call(request):
   reload(ai_module)
   from ai_module import img_path_list
   from ai_module import txt_path_list
   
   data = {
      "img_list" : img_path_list,
      "txt_list" : txt_path_list

   }

   img_path_list = []
   txt_path_list = []
   
   return JsonResponse(data)