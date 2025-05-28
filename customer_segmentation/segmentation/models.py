from django.db import models
from django.contrib.auth.models import User
import os
import json

# Create your models here.
class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    column_types = models.JSONField(default=dict)
    
    def __str__(self):
        return f"{self.name} - {self.user.username}"
    
    def delete(self, *args, **kwargs):
        # Delete the file when deleting the dataset
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        # Delete all clustering models and their results (and plots)
        for clustering_model in self.clustering_models.all():
            # Delete associated result and its plots
            if hasattr(clustering_model, 'result'):
                result = clustering_model.result
                # Delete visualization files
                if result.visualization_paths:
                    for path in result.visualization_paths.values():
                        abs_path = os.path.join(os.path.dirname(self.file.storage.path(self.file.name)), path) if not os.path.isabs(path) else path
                        if os.path.isfile(abs_path):
                            os.remove(abs_path)
                result.delete()
            clustering_model.delete()
        super().delete(*args, **kwargs)
        
class ClusteringModel(models.Model):
    ALGORITHM_CHOICES = [
        ('kmeans', 'K-Means'),
        ('hierarchical', 'Hierarchical Clustering'),
        ('dbscan', 'DBSCAN')
    ]
    
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='clustering_models')
    name = models.CharField(max_length=255)
    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES)
    parameters = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Store metrics
    metrics = models.JSONField(default=dict)
    
    def __str__(self):
        return f"{self.name} - {self.algorithm}"

class ClusteringResult(models.Model):
    model = models.OneToOneField(ClusteringModel, on_delete=models.CASCADE, related_name='result')
    num_clusters = models.IntegerField()
    cluster_distributions = models.JSONField(default=dict)
    cluster_profiles = models.JSONField(default=dict)
    
    # Path to stored visualizations
    visualization_paths = models.JSONField(default=dict)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Results for {self.model.name}"