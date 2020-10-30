from django.db import models

# Create your models here.

class User(models.Model):
    user_id = models.AutoField(db_column='User_ID', primary_key=True)
    name = models.CharField(db_column='Name',max_length=100)
    address =  models.CharField(db_column='Address',max_length=1000,null=True)
    sex =  models.CharField(db_column='Sex',max_length=10,blank=True,null=True)
    age =  models.CharField(db_column='Age',max_length=10,blank=True)
    job = models.CharField(db_column='Job',max_length=100,default="0")
    image = models.CharField(db_column='Image',max_length=100,default="0")
    created_at = models.DateTimeField(db_column='CreateTime',auto_now_add=True)
    updated_at = models.DateTimeField(db_column='UpdateTime',auto_now=True)