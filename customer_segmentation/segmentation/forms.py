from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Dataset, ClusteringModel

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file']
        
class ClusteringConfigForm(forms.ModelForm):
    n_clusters = forms.IntegerField(
        min_value=2, 
        max_value=20, 
        initial=3,
        help_text="Number of clusters for K-Means"
    )
    
    random_state = forms.IntegerField(
        initial=42,
        help_text="Random seed for reproducibility"
    )
    
    columns_to_use = forms.MultipleChoiceField(
        required=True,
        widget=forms.CheckboxSelectMultiple,
        help_text="Select columns to use for clustering"
    )
    
    class Meta:
        model = ClusteringModel
        fields = ['name', 'algorithm']
        
    def __init__(self, *args, column_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if column_choices:
            self.fields['columns_to_use'].choices = column_choices