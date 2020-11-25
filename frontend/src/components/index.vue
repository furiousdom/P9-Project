<template>
  <v-container>
    <v-row no-gutters>
      <v-col v-for="i in 24" :key="i" cols="3">
        <v-card height="500" outlined>
          <v-card-title class="blue-grey lighten-5">{{ drugSample.name }}</v-card-title>
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
import drugSample from '@/assets/drugSample';

export default {
  name: 'catalog',
  data: () => ({ drugSample })
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
