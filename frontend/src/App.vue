<template>
  <v-app id="inspire">
    <v-navigation-drawer v-model="drawer" app />
    <v-app-bar app>
      <v-app-bar-nav-icon @click="drawer = !drawer" />
      <v-toolbar-title>Application</v-toolbar-title>
    </v-app-bar>
    <v-main>
      <catalog />
    </v-main>
  </v-app>
</template>

<script>
import Catalog from './components';

export default {
  name: 'app',
  data: () => ({ drawer: null }),
  components: { Catalog }
};

</script>
