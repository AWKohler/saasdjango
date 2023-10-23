from django.urls import path
from api import views

from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()

urlpatterns = [
    path('bots/', include(router.urls)),

    path('', views.apiOverview, name="api-overview"),
    path('ai/', views.gpt, name="generate_names"),
    path('whoami/', views.get_user, name="whoami"),
    path('initbot/', views.init_bot, name="init-bot"),
    path('squery/', views.query_bot_s, name="query-bot-s"),
    path('query/', views.query_bot, name="query-bot"),

    path('bot/get/<int:bot_id>/', views.BotIDGetView.as_view(), name='bot_get'),
    path('bot/list/', views.BotListView.as_view(), name='bot_list'),
    path('bot/create/', views.BotCreateView.as_view(), name='bot_create'),
    path('bot/update/<int:bot_id>/', views.BotUpdateView.as_view(), name='bot_update'),
    path('bot/delete/<int:bot_id>/', views.BotDeleteView.as_view(), name='bot_delete'),
]