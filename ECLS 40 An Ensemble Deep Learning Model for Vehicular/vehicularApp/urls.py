from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("Signup.html", views.Signup, name="Signup"),
			path("SignupAction", views.SignupAction, name="SignupAction"),	    	
			path("UserLogin.html", views.UserLogin, name="UserLogin"),
			path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
			path("ProcessData", views.ProcessData, name="ProcessData"),
			path("TrainRF", views.TrainRF, name="TrainRF"),
			path("TrainLSTM", views.TrainEnsembleModel, name="TrainLSTM"),
			path("Predict", views.Predict, name="Predict"),
			path("PredictAction", views.PredictAction, name="PredictAction"),
]