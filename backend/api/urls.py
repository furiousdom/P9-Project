from django.urls import path
from .views import DrugView

urlpatterns = [
    path('home/', DrugView.as_view())
]
