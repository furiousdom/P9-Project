<template>
  <v-navigation-drawer
    permanent
    app
    color="grey lighten-2"
    width="320">
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
              :rules="proteinNameRules"
              :error-messages="proteinNameErrors"
              label="Name"
              required
              outlined />

            <v-btn @click="submit" :disabled="!valid" class="mr-4">
              submit
            </v-btn>
            <v-btn @click="reset">
              clear
            </v-btn>
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
    proteinNameRules: [name => !!name || 'Name is required'],
    proteinNameErrors: null
  }),
  methods: {
    validate() {
      this.$refs.form.validate();
    },
    reset() {
      this.$refs.form.reset();
    },
    resetValidation() {
      this.$refs.form.resetValidation();
    },
    submit() {
      const { proteinName } = this;
      api.search({ proteinName })
        .then(({ data }) => {
          this.$emit('submit', data);
        });
    }
  }
};
</script>

<style lang="scss" scoped>
.title {
  margin: 1rem 0 1rem 0;
}
</style>
