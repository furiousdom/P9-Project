<template>
  <v-navigation-drawer color="grey lighten-2" width="320" permanent app>
    <v-list>
      <v-list-item>
        <v-list-item-content>
          <v-list-item-title class="title">
            CS-IT9 Project
          </v-list-item-title>
        </v-list-item-content>
      </v-list-item>
      <v-divider />
      <v-list-item>
        <v-list-item-content>
          <v-form ref="form" v-model="valid">
            <v-text-field
              v-model="proteinName"
              :rules="rules.proteinName"
              label="Name"
              required outlined />
            <v-text-field
              v-model.number="noResults"
              :rules="rules.noResults"
              type="number"
              label="No. of Results" />
            <v-checkbox v-model="logging" label="Enable Logging" />
            <v-btn @click="submit" :disabled="!valid" class="mr-4">submit</v-btn>
            <v-btn @click="reset">clear</v-btn>
          </v-form>
        </v-list-item-content>
      </v-list-item>
    </v-list>
  </v-navigation-drawer>
</template>

<script>
import api from '@/services/drugs';

export default {
  name: 'sidebar',
  data: () => ({
    valid: true,
    proteinName: '',
    logging: false,
    noResults: 1,
    rules: {
      proteinName: [name => !!name || 'Name is required.'],
      noResults: [val => val > 0 || 'Must be a non-negative number.']
    }
  }),
  methods: {
    validate() {
      this.$refs.form.validate();
    },
    reset() {
      this.$refs.form.reset();
      this.logging = false;
    },
    resetValidation() {
      this.$refs.form.resetValidation();
    },
    async submit() {
      const { proteinName, noResults, logging } = this;
      const { data } = await api.search({ proteinName, noResults, logging });
      this.$emit('submit', data);
    }
  }
};
</script>

<style lang="scss" scoped>
.title {
  margin: 1rem 0 1rem 0;
}
</style>
