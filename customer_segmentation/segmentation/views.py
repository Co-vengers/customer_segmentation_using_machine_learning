import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import io
import base64
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.urls import reverse
from .forms import UserRegistrationForm, DatasetUploadForm, ClusteringConfigForm
from .models import Dataset, ClusteringModel, ClusteringResult
from .ml.preprocessor import preprocess_data
from .ml.clustering import perform_clustering
from django.utils.safestring import mark_safe
import json



def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful!")
            return redirect('dashboard')
        else:
            messages.error(request, "Registration failed. Please correct the errors.")
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('login')  # Or wherever you want to redirect after logout

@login_required(login_url='login')
def dashboard_view(request):
    datasets = Dataset.objects.filter(user=request.user).order_by('-uploaded_at')
    clustering_models = ClusteringModel.objects.filter(
        dataset__user=request.user
    ).select_related('dataset', 'result').order_by('-created_at')
    
    return render(request, 'segmentation/dashboard.html', {
        'datasets': datasets,
        'clustering_models': clustering_models
    })

@login_required
def upload_dataset_view(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.user = request.user
            
            # Save the dataset first to have access to the file
            dataset.save()
            
            # Read and analyze the file to get metadata
            try:
                file_path = dataset.file.path
                df = pd.read_csv(file_path)
                
                # Update dataset metadata
                dataset.rows = len(df)
                dataset.columns = len(df.columns)
                
                # Store column types
                column_types = {}
                for column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        column_types[column] = 'numeric'
                    else:
                        column_types[column] = 'categorical'
                
                dataset.column_types = column_types
                dataset.save()
                
                # Store the preview data in session for the next view
                preview_data = df.head(10).to_dict(orient='records')
                request.session['preview_data'] = preview_data
                request.session['dataset_id'] = dataset.id
                
                messages.success(request, f"Dataset '{dataset.name}' uploaded successfully!")
                return redirect('preview_dataset', dataset_id=dataset.id)
            
            except Exception as e:
                # Delete the dataset if there's an error processing it
                dataset.delete()
                messages.error(request, f"Error processing dataset: {str(e)}")
                return redirect('upload_dataset')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'segmentation/upload.html', {'form': form})

@login_required
def preview_dataset_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    
    try:
        df = pd.read_csv(dataset.file.path)
        preview_data = df.head(10).to_dict(orient='records')
        column_names = df.columns.tolist()
        
        # Calculate basic statistics for numeric columns
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isna().sum()
                }
        
        context = {
            'dataset': dataset,
            'preview_data': preview_data,
            'column_names': column_names,
            'stats': stats,
            'column_names_json': mark_safe(json.dumps(column_names)),
            'preview_data_json': mark_safe(json.dumps(preview_data)),
        }
        
        return render(request, 'segmentation/preview.html', context)

    except Exception as e:
        messages.error(request, f"Error previewing dataset: {str(e)}")
        return redirect('dashboard')


# Add this function to your views

@login_required
def process_clustering_view(request, model_id):
    # Retrieve the clustering model instance
    model = get_object_or_404(
        ClusteringModel,
        id=model_id,
        dataset__user=request.user  # Ensure the model belongs to the user
    )
    dataset = model.dataset

    try:
        # Read the dataset file
        df = pd.read_csv(dataset.file.path)

        # Get the columns to use from model parameters
        columns_to_use = model.parameters.get('columns_to_use', [])
        df = df[columns_to_use]

        # Preprocess the data (ensure all columns are numeric after this step)
        processed_df, _ = preprocess_data(df, columns_to_use)

        # Perform clustering using the selected algorithm and parameters
        algorithm = model.algorithm
        parameters = model.parameters

        # Call the clustering function
        labels, metrics, cluster_profiles = perform_clustering(
            processed_df,
            algorithm,
            parameters.get('n_clusters', 3),
            parameters.get('random_state', 42)
        )

        # Generate PCA visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(processed_df)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = labels

        # Create and save PCA plot (use non-interactive backend)
        import matplotlib
        matplotlib.use('Agg')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
        plt.title('PCA Plot of Clusters')

        # Save the plot to media directory
        plot_filename = f'pca_{model.id}.png'
        plot_path = os.path.join(settings.MEDIA_ROOT, 'clustering_plots', plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        # Create ClusteringResult entry
        # Convert all cluster_profiles values to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        cluster_profiles_serializable = convert(cluster_profiles)

        result = ClusteringResult.objects.create(
            model=model,
            num_clusters=int(parameters.get('n_clusters', 3)),
            cluster_distributions={int(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
            cluster_profiles=cluster_profiles_serializable,
            visualization_paths={'pca_plot': os.path.join('clustering_plots', plot_filename)}
        )

        messages.success(request, "Clustering completed successfully!")
        return redirect('clustering_results', result_id=result.id)

    except Exception as e:
        messages.error(request, f"Error during clustering processing: {str(e)}")
        return redirect('dashboard')

@login_required
def configure_clustering_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    
    try:
        df = pd.read_csv(dataset.file.path)
        column_choices = [(col, col) for col in df.columns]
        
        if request.method == 'POST':
            form = ClusteringConfigForm(request.POST, column_choices=column_choices)
            if form.is_valid():
                clustering_model = form.save(commit=False)
                clustering_model.dataset = dataset
                
                # Store parameters
                parameters = {
                    'n_clusters': form.cleaned_data['n_clusters'],
                    'random_state': form.cleaned_data['random_state'],
                    'columns_to_use': form.cleaned_data['columns_to_use']
                }
                clustering_model.parameters = parameters
                clustering_model.save()
                
                # Redirect to processing view
                return redirect('process_clustering', model_id=clustering_model.id)
        else:
            form = ClusteringConfigForm(column_choices=column_choices)
        
        return render(request, 'segmentation/configure_clustering.html', {
            'form': form,
            'dataset': dataset
        })
    except Exception as e:
        messages.error(request, f"Error configuring clustering: {str(e)}")
        return redirect('dashboard')


@login_required
def clustering_results_view(request, result_id):
    result = get_object_or_404(
        ClusteringResult, 
        id=result_id, 
        model__dataset__user=request.user
    )
    model = result.model
    dataset = model.dataset
    
    return render(request, 'segmentation/results.html', {
        'result': result,
        'model': model,
        'dataset': dataset,
        'MEDIA_URL': settings.MEDIA_URL
    })

@login_required
def download_report_view(request, result_id):
    result = get_object_or_404(
        ClusteringResult, 
        id=result_id, 
        model__dataset__user=request.user
    )
    
    # Generate a PDF report or CSV export
    # Implementation will depend on chosen libraries (e.g., ReportLab for PDF)
    # For simplicity, we'll just export cluster profiles as CSV
    
    model = result.model
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{model.name}_report.csv"'
    
    # Convert cluster profiles to DataFrame and export
    profiles_data = []
    for cluster_id, profile in result.cluster_profiles.items():
        row = {'cluster_id': cluster_id}
        row.update(profile)
        profiles_data.append(row)
    
    profiles_df = pd.DataFrame(profiles_data)
    profiles_df.to_csv(response, index=False)
    
    return response

@login_required
def delete_dataset_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, user=request.user)
    
    if request.method == 'POST':
        dataset_name = dataset.name
        dataset.delete()
        messages.success(request, f"Dataset '{dataset_name}' deleted successfully.")
        return redirect('dashboard')
    
    return render(request, 'segmentation/delete_confirm.html', {'dataset': dataset})

