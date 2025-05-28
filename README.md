# Customer Segmentation Using Machine Learning

This project is a Django-based web application for customer segmentation using machine learning techniques. It allows users to upload customer datasets, configure clustering algorithms, visualize results, and manage segmentation workflows through an interactive dashboard.

## Features
- Upload and preview customer datasets (CSV format)
- Configure and run clustering algorithms (e.g., K-Means, PCA)
- Visualize clustering results with plots
- User authentication (login/register)
- Dashboard for managing segmentation tasks

## Project Structure
- `customer_segmentation/` - Django project settings and configuration
- `segmentation/` - Main app for segmentation logic, views, templates, and ML code
  - `ml/` - Machine learning scripts (clustering, preprocessing)
  - `templates/` - HTML templates for the web interface
  - `static/` - Static files (CSS, JS, images)
- `datasets/` - Example datasets
- `media/` - Uploaded files and generated plots
- `clustering_plots/` - Pre-generated clustering visualizations

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd customer_segmentation_using_machine_learning
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Apply migrations:
   ```bash
   python manage.py migrate
   ```
5. Create a superuser (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```
6. Run the development server:
   ```bash
   python manage.py runserver
   ```
7. Access the app at [http://localhost:8000/](http://localhost:8000/)

## Usage
- Register or log in to your account.
- Upload a customer dataset (CSV).
- Configure clustering parameters and run segmentation.
- View and download clustering results and visualizations.

## Machine Learning
- Clustering algorithms and preprocessing are implemented in `segmentation/ml/`.
- Visualizations are saved in the `media/clustering_plots/` directory.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Built with [Django](https://www.djangoproject.com/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Data visualization with [matplotlib](https://matplotlib.org/)
