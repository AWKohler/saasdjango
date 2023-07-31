from django.urls import path
from api import views

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BotViewSet

router = DefaultRouter()
router.register(r'bots', BotViewSet)

urlpatterns = [
    path('bots/', include(router.urls)),

    path('', views.apiOverview, name="api-overview"),
    path('ai/', views.generate_names, name="generate_names"),
    path('falcon/', views.falcon, name="falcon"),
    path('whoami/', views.get_user, name="whoami"),
    path('initbot/', views.init_bot, name="init-bot"),
    path('squery/', views.query_bot_s, name="query-bot-s"),
    # path('pdfs/', views.pdf_upload, name='pdf_upload'),
    # path('pdfs/<int:id>/', views.pdf_delete, name='pdf_delete'),
    # path('pdfs/list/', views.pdf_list, name='pdf_list'),
]