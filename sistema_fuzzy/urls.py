# sistema_fuzzy/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Página principal
    path('', views.Home_fuzzy, name='dashboard'),
    
    # Nueva ruta para el modelo fuzzy completo
    path('fuzzy-model/', views.fuzzy_model_complete, name='fuzzy_model_complete'),
    
    # Funciones de membresía
    path('membership-functions/', views.membership_functions, name='membership_functions'),
    path('membership-functions/add/', views.add_membership_function, name='add_membership_function'),
    path('membership-functions/edit/<int:id>/', views.edit_membership_function, name='edit_membership_function'),
    path('membership-functions/delete/<int:id>/', views.delete_membership_function, name='delete_membership_function'),
    
    # Simulación
    path('simulation/', views.simulation, name='simulation'),
    
    # Reglas difusas
    path('fuzzy-rules/', views.fuzzy_rules, name='fuzzy_rules'),
    
    # Análisis y estadísticas
    path('analytics/', views.analytics, name='analytics'),
    path('api/stats/', views.api_stats, name='api_stats'),
    
    # Gestión y configuración
    # En tu urls.py
    path('settings/', views.system_settings, name='settings'),  # Cambiar aquí también
    path('data-management/', views.data_management, name='data_management'),
    path('export-data/', views.export_data, name='export_data'),
    
    # Información y ayuda
    path('about-mamdani/', views.about_mamdani, name='about_mamdani'),
    path('performance/', views.performance, name='performance'),
    path('tutorial/', views.tutorial, name='tutorial'),
    path('help/', views.help, name='help'),
    
    # Actividad y logs
    path('activity-log/', views.activity_log, name='activity_log'),
]
