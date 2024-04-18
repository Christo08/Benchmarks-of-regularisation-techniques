import json


def create_numeric_run_object(method, seed, startTime, endTime, settings, training_losses, training_accuracies,
                              testing_losses, testing_accuracies, validation_losses, validation_accuracies):
    run_object = {
        "method": method,
        "seed": seed,
        "runtime": format_runtime(startTime,endTime),
        "settings": settings.to_json_serializable(),
        "results": {
            "accuracies":{
                "training": training_accuracies,
                "testing": testing_accuracies,
                "validation": validation_accuracies
            },
            "losses":{
                "training": training_losses,
                "testing": testing_losses,
                "validation": validation_losses,
            }
        }
    }
    return run_object

def format_runtime(startTime, endTime):
    seconds = endTime - startTime
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
