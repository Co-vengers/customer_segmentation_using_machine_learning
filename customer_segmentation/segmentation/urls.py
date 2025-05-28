from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    
    # Main application URLs
    path('', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_dataset_view, name='upload_dataset'),
    path('preview/<int:dataset_id>/', views.preview_dataset_view, name='preview_dataset'),
    path('configure/<int:dataset_id>/', views.configure_clustering_view, name='configure_clustering'),
    path('process/<int:model_id>/', views.process_clustering_view, name='process_clustering'),
    path('results/<int:result_id>/', views.clustering_results_view, name='clustering_results'),
    path('download/<int:result_id>/', views.download_report_view, name='download_report'),
    path('delete/<int:dataset_id>/', views.delete_dataset_view, name='delete_dataset'),
]