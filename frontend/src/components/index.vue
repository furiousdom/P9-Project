<template>
  <v-container>
    <v-row no-gutters>
      <v-col v-for="drug in drugs" :key="drug.primary_id" cols="3">
        <v-card height="500" outlined>
          <v-card-title class="blue-grey lighten-5">
            <v-tooltip bottom>
              <template v-slot:activator="{ on, attrs }">
                <span v-on="on" v-bind="attrs">{{ truncate(drug.name, 19) }}</span>
              </template>
              <span>{{ drug.name }}</span>
            </v-tooltip>
          </v-card-title>
          <v-img
            max-height="150"
            max-width="150"
            src="@/assets/chem-structure-sample.webp" />
          <v-row class="pa-4" no-gutters>
            <v-col v-for="propName in Object.keys(drugSample.properties)" :key="propName" cols="6">
              <v-list-item two-line>
                <v-list-item-content>
                  <v-list-item-title>{{ propName }}</v-list-item-title>
                  <v-list-item-subtitle>{{ drugSample.properties[propName] }}</v-list-item-subtitle>
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
    truncate: (name, length) => truncate(name, { length })
  },
  async mounted() {
    const { data } = await api.fetch();
    this.drugs = data;
  }
};
</script>

<style lang="scss" scoped>
.v-sheet.v-card {
  border-radius: 0;

  .v-image {
    margin: 1rem auto;
  }
}
</style>
