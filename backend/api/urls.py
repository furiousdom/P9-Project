from django.urls import path
from .views import DrugView

urlpatterns = [
    path('drugs/', DrugView.as_view())
]
