# sistema_fuzzy/admin.py
from django.contrib import admin
from .models import Variable, MembershipFunction

admin.site.register(Variable)
admin.site.register(MembershipFunction)
