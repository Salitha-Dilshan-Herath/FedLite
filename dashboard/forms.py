from django import forms

class ExperimentForm(forms.Form):

    dataset = forms.ChoiceField(
        choices=[
            ("mnist", "MNIST"),
            ("fashion", "Fashion-MNIST"),
            ("cifar10", "CIFAR-10"),
        ],
        initial="cifar10"
    )

    rounds = forms.IntegerField(min_value=1, max_value=200, initial=30)
    num_clients = forms.IntegerField(min_value=2, max_value=200, initial=20)
    fraction_fit = forms.FloatField(min_value=0.05, max_value=1.0, initial=0.2)
    alpha = forms.FloatField(min_value=0.05, max_value=5.0, initial=0.5)

    topk_percent = forms.FloatField(min_value=0.1, max_value=50.0, initial=5.0)
    quant_bits = forms.ChoiceField(
        choices=[("8", "8-bit"), ("4", "4-bit")],
        initial="8"
    )
    local_epochs = forms.IntegerField(min_value=1, max_value=10, initial=2)
    error_feedback = forms.BooleanField(required=False, initial=True)