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
          <form>
            <v-text-field
              v-model="proteinName"
              :error-messages="nameErrors"
              label="Name"
              outlined
              required />

            <v-btn @click="submit" class="mr-4">
              submit
            </v-btn>
            <v-btn @click="clear">
              clear
            </v-btn>
          </form>
        </v-list-item-content>
      </v-list-item>
    </v-list>
  </v-navigation-drawer>
</template>

<script>
import api from '@/services/drugs';

export default {
  name: 'drawer',
  data: () => ({ proteinName: '', nameErrors: null }),
  methods: {
    clear() {
      this.proteinName = '';
    },
    submit() {
      const proteinName = this.proteinName;
      if (proteinName === '') this.nameErrors = 'The protein must have a name.';
      else api.search({ proteinName }).then(({ data }) => console.log(data));
    }
  }
};
</script>

<style lang="scss" scoped>
.title {
  margin: 1rem 0 1rem 0;
}
</style>
