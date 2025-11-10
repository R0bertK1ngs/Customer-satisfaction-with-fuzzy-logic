# sistema_fuzzy/models.py
from django.db import models

class Variable(models.Model):
    VARIABLE_TYPES = (
        ('input', 'Entrada'),
        ('output', 'Salida'),
    )

    name = models.CharField(max_length=100, unique=True)
    var_type = models.CharField(max_length=10, choices=VARIABLE_TYPES)

    def __str__(self):
        return f"{self.name} ({self.get_var_type_display()})"

class MembershipFunction(models.Model):
    FUNCTION_TYPES = (
        ('triangular', 'Triangular'),
        ('trapezoidal', 'Trapezoidal'),
        ('gaussian', 'Gaussiana'),
        ('singleton', 'Singleton'),
    )

    variable = models.ForeignKey(Variable, on_delete=models.CASCADE, related_name='membership_functions')
    name = models.CharField(max_length=100)
    func_type = models.CharField(max_length=20, choices=FUNCTION_TYPES)
    parameters = models.JSONField(help_text='Parámetros: ej. [a, b, c] para triangular')

    def __str__(self):
        return f"{self.name} ({self.func_type}) - {self.variable.name}"
