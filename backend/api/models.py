# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AtcCodesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    atc_codes = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'atc_codes_table'


class CategoriesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    category = models.TextField(blank=True, null=True)
    mesh_id = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'categories_table'


class DosagesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    form = models.TextField(blank=True, null=True)
    route = models.TextField(blank=True, null=True)
    strength = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dosages_table'


class DrugInteractionsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id_1 = models.CharField(max_length=20, blank=True, null=True)
    drug_id_2 = models.CharField(max_length=20, blank=True, null=True)
    sd_name = models.TextField(blank=True, null=True)
    sd_desc = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'drug_interactions_table'


class GroupsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    primary_id = models.CharField(max_length=20, blank=True, null=True)
    groups = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'groups_table'


class MixturesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    mixture_name = models.TextField(blank=True, null=True)
    ingredient = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mixtures_table'


class PathwaysTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    pathways = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'pathways_table'


class ProductsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    product_name = models.TextField(blank=True, null=True)
    labeller = models.TextField(blank=True, null=True)
    ndc_id = models.TextField(blank=True, null=True)
    ndc_product_code = models.TextField(blank=True, null=True)
    dpd_id = models.TextField(blank=True, null=True)
    ema_product_code = models.TextField(blank=True, null=True)
    ema_ma_number = models.TextField(blank=True, null=True)
    started_marketing_on = models.DateField(blank=True, null=True)
    ended_marketing_on = models.DateField(blank=True, null=True)
    dosage_form = models.TextField(blank=True, null=True)
    strength = models.TextField(blank=True, null=True)
    route = models.TextField(blank=True, null=True)
    fda_application_number = models.TextField(blank=True, null=True)
    generic = models.BooleanField(blank=True, null=True)
    over_the_counter = models.BooleanField(blank=True, null=True)
    approved = models.BooleanField(blank=True, null=True)
    country = models.CharField(max_length=20, blank=True, null=True)
    source = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'products_table'


class SecondaryIdsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    primary_id = models.CharField(max_length=20, blank=True, null=True)
    secondary_id = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'secondary_ids_table'


class SequencesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    sequence_t = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sequences_table'


class SynonymsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    synonym_name = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'synonyms_table'


class TargetsTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug_id = models.CharField(max_length=20, blank=True, null=True)
    targets = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'targets_table'


class MainTable(models.Model):
    primary_id = models.CharField(primary_key=True, max_length=20)
    name = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    cas_number = models.CharField(max_length=20, blank=True, null=True)
    unii = models.CharField(max_length=20, blank=True, null=True)
    state = models.TextField(blank=True, null=True)
    indication = models.TextField(blank=True, null=True)
    pharmacodynamics = models.TextField(blank=True, null=True)
    mechanism = models.TextField(blank=True, null=True)
    toxicity = models.TextField(blank=True, null=True)
    metabolism = models.TextField(blank=True, null=True)
    absorbtion = models.TextField(blank=True, null=True)
    halflife = models.TextField(blank=True, null=True)
    protein_binding = models.TextField(blank=True, null=True)
    route_of_elimination = models.TextField(blank=True, null=True)
    volume_of_distribution = models.TextField(blank=True, null=True)
    clearance = models.TextField(blank=True, null=True)
    classification = models.TextField(blank=True, null=True)  # This field type is a guess.
    fda_label = models.TextField(blank=True, null=True)
    msds = models.TextField(blank=True, null=True)
    reactions = models.TextField(blank=True, null=True)  # This field type is a guess.
    snp_effects = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'main_table'


class PropertiesTable(models.Model):
    pkey = models.AutoField(primary_key=True)
    drug = models.OneToOneField(max_length=20, unique=True, to=MainTable, related_name='props', on_delete=models.CASCADE)
    properties = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'properties_table'

    def __str__(self):
        return '%s' % (self.properties)
