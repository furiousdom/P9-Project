<template>
  <v-container>
    <v-row no-gutters>
      <v-col v-for="drug in drugs" :key="drug.primary_id" cols="3">
        <v-card height="500" outlined>
          <v-card-title class="d-flex align-center blue-grey lighten-5">
            <v-badge :content="drugs.indexOf(drug) + 1" color="indigo" />
            <v-tooltip bottom>
              <template v-slot:activator="{ on, attrs }">
                <span v-on="on" v-bind="attrs">{{ truncate(drug.name, 17) }}</span>
              </template>
              <span>{{ drug.name }}</span>
            </v-tooltip>
          </v-card-title>
          <v-img
            max-height="150"
            max-width="150"
            src="@/assets/chem-structure-sample.webp" />
          <v-row no-gutters class="pa-4">
            <v-col v-for="prop in drug.eprops" :key="prop.kind" cols="6">
              <v-list-item two-line>
                <v-list-item-content>
                  <v-list-item-title>{{ prop.kind }}</v-list-item-title>
                  <v-list-item-subtitle>{{ prop.value }}</v-list-item-subtitle>
                </v-list-item-content>
              </v-list-item>
            </v-col>
          </v-row>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import api from '@/services/drugs';
import drugSample from '@/assets/drugSample';
import truncate from 'lodash/truncate';

export default {
  name: 'catalog',
  data: () => ({
    drugSample,
    drugs: null
  }),
  methods: {
    truncate: (name, length) => truncate(name, { length }),
    parseJsonProps(drugs) {
      drugs.forEach(drug => {
        const cprops = JSON.parse(drug.cprops)['calculated-properties'];
        const eprops = JSON.parse(drug.eprops)['experimental-properties'];
        if (cprops) drug.cprops = cprops.property;
        else drug.cprops = null;
        if (eprops) drug.eprops = eprops.property;
      });
    }
  },
  async mounted() {
    const { data } = await api.fetch();
    this.parseJsonProps(data);
    this.drugs = data;
  }
};
</script>

<style lang="scss" scoped>
::v-deep .v-badge {
  margin: 0.625rem 1.5rem 0 0;
}
.v-sheet.v-card {
  border-radius: 0;

  .v-image {
    margin: 1rem auto;
  }
}
</style>
