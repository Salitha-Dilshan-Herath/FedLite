from django.urls import path
from .views import index, start_run, progress

urlpatterns = [
    path("", index, name="index"),
    path("start/", start_run, name="start_run"),
    path("progress/<str:job_id>/", progress, name="progress"),
]