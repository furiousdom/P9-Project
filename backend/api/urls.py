from django.urls import path
from .views import DrugView, Drugs, MakeMolGraphs

urlpatterns = [
    path('drugs/', DrugView.as_view()),
    path('drugs/search/', Drugs.as_view()),
    path('drugs/makeimages', MakeMolGraphs.as_view())
]
