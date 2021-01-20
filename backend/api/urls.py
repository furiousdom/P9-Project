from django.urls import path
from .views import DrugView, Drugs, MakeSmiFile

urlpatterns = [
    path('drugs/', DrugView.as_view()),
    path('drugs/search/', Drugs.as_view()),
    path('makesmi/', MakeSmiFile.as_view())
]
