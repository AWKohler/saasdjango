from rest_framework import serializers
from .models import Bot, Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ('id', 'file', 'uploaded_at')


class BotSerializer(serializers.ModelSerializer):
    documents = DocumentSerializer(many=True, read_only=True)

    class Meta:
        model = Bot
        fields = ('id', 'user', 'name', 'use', 'documents')