from django.urls import path
from .views import DrugXmlView

urlpatterns = [
    path('xmldrugs', DrugXmlView.as_view())
]
