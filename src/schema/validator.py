from jsonschema import validate, ValidationError
import json

with open("src/schema/dilemma_schema.json") as f:
    SCHEMA = json.load(f)

def validate_dilemma(d):
    try:
        validate(instance=d, schema=SCHEMA)
        return True
    except ValidationError as e:
        return False, str(e)
